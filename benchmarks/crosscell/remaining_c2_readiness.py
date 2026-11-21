"""Audit scFoundation and PRESCRIBE against the fixed C2 experiment contract."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import anndata
import numpy as np
import torch

from benchmarks.common.study_fit_partition import crosscell_fit_rows
from benchmarks.crosscell.official_fit_probe import PRIOR_SHA, SUPPORT_SHA
from benchmarks.crosscell.scbig_c2_flow_fit import HGNC, HGNC_SHA
from benchmarks.crosscell.study_data import cohort_inputs
from benchmarks.crosscell.study_paths import SERVER, STUDY
from gradpert.data._io import atomic_json, read_json
from gradpert.hashing import sha256_file

UPSTREAMS = SERVER / "upstreams"
PRESCRIBE_PROBE = STUDY / "prescribe-crosscell-c2-fit-probe-a3"
PRESCRIBE_RECEIPT_SHA = "8926409696666b2769a97189c88351d20aefe67993beea50ba22680200617f7d"
FOUNDATION_WEIGHTS = UPSTREAMS / "scfoundation-c2-study-assets/models.genbio-cb43415.ckpt"
FOUNDATION_WEIGHTS_SHA = "9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9"


def hgnc_links():
    if sha256_file(HGNC) != HGNC_SHA:
        raise ValueError("C2 HGNC reference changed")
    links = defaultdict(list)
    with HGNC.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            official = row["symbol"]
            previous = set(row["prev_symbol"].split("|")) - {""}
            aliases = set(row["alias_symbol"].split("|")) - {""}
            family = previous | aliases | {official}
            for name in family:
                tier = 0 if name == official else 1 if name in previous else 2
                links[name].append((tier, official, family))
    return links


def reconcile(names, vocabulary, links):
    claimed = defaultdict(list)
    mapping = {}
    missing = []
    for name in sorted(set(names)):
        if name in vocabulary:
            resolved = name
        else:
            matches = links.get(name, ())
            if not matches:
                missing.append(name)
                continue
            best = min(tier for tier, _, _ in matches)
            choices = set()
            for tier, approved, family in matches:
                if tier == best:
                    choices |= {approved} if approved in vocabulary else vocabulary & family
            if len(choices) != 1:
                missing.append(name)
                continue
            (resolved,) = choices
        mapping[name] = resolved
        claimed[resolved].append(name)
    return {
        "canonical_names": len(set(names)),
        "mapped_names": len(mapping),
        "aliases": {key: value for key, value in mapping.items() if key != value},
        "unresolved": missing,
        "collisions": {key: names for key, names in claimed.items() if len(names) > 1},
    }


def scfoundation(panel, test, links, vocab):
    vocabulary = vocab_names(vocab)
    return {
        "checkpoint_path": str(FOUNDATION_WEIGHTS),
        "expected_checkpoint_sha256": FOUNDATION_WEIGHTS_SHA,
        "checkpoint_present": FOUNDATION_WEIGHTS.is_file(),
        "checkpoint_sha256": sha256_file(FOUNDATION_WEIGHTS)
        if FOUNDATION_WEIGHTS.is_file()
        else None,
        "vocabulary_sha256": sha256_file(vocab),
        "expression": reconcile(panel, vocabulary, links),
        "test": reconcile(test, vocabulary, links),
    }


def vocab_names(path):
    with path.open() as stream:
        return {row["gene_name"] for row in csv.DictReader(stream, delimiter="\t")}


def prescribe(training, test, links):
    from src.nn.loss import ListMLELoss

    path = UPSTREAMS / "prescribe-c2-study-assets/scgpt-primary.pkl"
    if sha256_file(path) != PRIOR_SHA["prescribe"]:
        raise ValueError("PRESCRIBE released scGPT prior changed")
    with path.open("rb") as stream:
        vocabulary = set(pickle.load(stream))
    if sha256_file(PRESCRIBE_PROBE / "result.json") != PRESCRIBE_RECEIPT_SHA:
        raise ValueError("the PRESCRIBE PCA probe receipt changed")
    pca_path = PRESCRIBE_PROBE / "fit_pca.npz"
    with np.load(pca_path, allow_pickle=False) as pca:
        pca_width = int(pca["components"].shape[1])
        pca_rank = int(pca["components"].shape[0])
    objective = ListMLELoss()
    scores = torch.tensor([0.25, -4.0, 0.9, 2.0], dtype=torch.float64, requires_grad=True)
    first = objective(scores, torch.tensor([0, 1, 2, 3]))
    second = objective(scores, torch.tensor([3, 2, 1, 0]))
    (gradient,) = torch.autograd.grad(first, scores)
    (reversed_gradient,) = torch.autograd.grad(second, scores)
    return {
        "released_prior_sha256": PRIOR_SHA["prescribe"],
        "training": reconcile(training, vocabulary, links),
        "test": reconcile(test, vocabulary, links),
        "prior_fit_pca_sha256": sha256_file(pca_path),
        "prior_fit_pca_width": pca_width,
        "prior_fit_pca_rank": pca_rank,
        "rank_probe": {
            "loss_ascending": float(first.detach()),
            "loss_descending": float(second.detach()),
            "absolute_loss_difference": float((first - second).abs().detach()),
            "maximum_gradient_difference": float((gradient - reversed_gradient).abs().max()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent.resolve() != STUDY or args.output.exists():
        raise ValueError("provide a fresh C2 server receipt path")
    source, _, cohort, frozen = cohort_inputs()
    support = STUDY / "frozen/crosscell-c2-shared-fit-support-a1.json"
    if sha256_file(support) != SUPPORT_SHA:
        raise ValueError("C2 shared support changed")
    adata = anndata.read_h5ad(source, backed="r")
    try:
        panel = list(map(str, adata.var_names))
        conditions = adata.obs["condition"].astype(str).to_numpy()
        roles = crosscell_fit_rows(frozen, conditions, cohort, compatibility=read_json(support))
        training = {
            condition.removesuffix("+ctrl")
            for condition in conditions[np.r_[roles["train"], roles["val"]]]
        }
        test = {condition.removesuffix("+ctrl") for condition in cohort["condition_ids"]}
        count_metadata = [str(key) for key in adata.obs if "count" in str(key).lower()]
    finally:
        adata.file.close()
    if len(panel) != 3352 or len(training) != 1957 or len(test) != 128 or training & test:
        raise ValueError("unexpected frozen C2 identities")
    links = hgnc_links()
    vocab = UPSTREAMS / "scfoundation-c2-study/OS_scRNA_gene_index.19264.tsv"
    atomic_json(
        args.output,
        {
            "status": "metadata_and_released_loss_audited",
            "scenario": "C2",
            "formal_eligible": False,
            "training_updates": 0,
            "expression_matrices_materialized": False,
            "test_expression_rows_read": 0,
            "count_metadata_columns": count_metadata,
            "scfoundation": scfoundation(panel, test, links, vocab),
            "prescribe": prescribe(training, test, links),
        },
    )


if __name__ == "__main__":
    main()
