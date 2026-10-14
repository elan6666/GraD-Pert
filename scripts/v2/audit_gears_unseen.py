"""Read-only sealed GEARS replay: decoder, masked gradients and Pearson inputs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.common.official_checkout import official_module_session
from benchmarks.expression_holdout.config import load_config, require_evaluation_identity
from benchmarks.expression_holdout.contract import FittedAdapter
from benchmarks.expression_holdout.gears import OFFICIAL_SHA, masked_official_loss, predict
from gradpert.data._io import atomic_json, read_json
from gradpert.evaluation.data import CanonicalEvaluationData
from gradpert.evaluation.metrics import compute_condition_metrics_v2, pearson_correlation
from gradpert.evaluation.state import load_evaluation_state
from gradpert.execution.identity import inspect_source_identity
from gradpert.hashing import sha256_file, sha256_json


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "max_abs": float(np.abs(values).max()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "std": float(values.std()),
        "exact_zero_count": int((values == 0).sum()),
        "count": int(values.size),
    }


def audit(args):
    import anndata
    import pandas as pd
    import torch
    from run_group import lock
    from run_unified_group import ALLOCATOR, LEASE_ROOT
    from shared_gpu import admit, snapshot

    root = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    if not run.is_relative_to("/data/yilangliu") or not args.output.resolve().is_relative_to(
        "/data/yilangliu"
    ):
        raise ValueError("checkpoint, data and diagnostics are server-only")
    if args.output.exists():
        raise FileExistsError("fresh diagnostic output required")
    if os.environ.get("PYTORCH_ALLOC_CONF") != ALLOCATOR:
        raise ValueError("allocator contract missing")
    identity = inspect_source_identity(
        root,
        formal=True,
        expected_repository="https://github.com/elan6666/GraD-Pert.git",
        publication_receipt=args.publication,
        expected_publication_receipt_sha256=args.publication_sha256,
    )
    cfg, rt = load_config(args.config), read_json(args.runtime)
    receipt, fit, terminal = (
        read_json(run / name) for name in ("receipt.json", "fit-receipt.json", "last-test.json")
    )
    checkpoint = run / "fit/last.pt"
    before = sha256_file(checkpoint)
    assert before == receipt["checkpoint_sha256"] == terminal["checkpoint_sha256"]
    assert sha256_file(run / "last-test.json") == receipt["last_test_sha256"]
    assert sha256_file(args.config) == receipt["config_sha256"]
    assert sha256_file(args.runtime) == receipt["runtime_sha256"]
    assert receipt["scientific_complete"] and receipt["zero_pkl"]
    assert cfg["source_code"]["commit"] == receipt["official_commit"] == OFFICIAL_SHA
    assert cfg["model"]["no_perturb"] is False
    torch.set_num_threads(2)
    started = time.monotonic()
    result = {
        "kind": "gears-unseen-read-only-audit-1",
        "audit_source": identity.payload(),
        "training_source": receipt["source_commit"],
        "historical_evaluation_source": terminal["source_commit"],
        "checkpoint_sha256": before,
        "official_commit": OFFICIAL_SHA,
        "config_sha256": receipt["config_sha256"],
        "runtime_sha256": receipt["runtime_sha256"],
        "optimizer_updates": 0,
        "no_original_artifact_write": True,
    }
    policy = read_json(rt["policy"])
    with contextlib.ExitStack() as stack:
        if args.device == "cuda:0":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
                raise ValueError("audit CUDA device must be exact physical GPU0")
            stack.enter_context(lock(LEASE_ROOT / "v2-gpu-0.lock"))
            admit(policy, snapshot(os.getpid()))
            torch.cuda.set_per_process_memory_fraction(policy["max_own_memory_fraction"], 0)
        stack.enter_context(
            official_module_session(
                checkout_root=rt["official_checkout"],
                expected_commit=OFFICIAL_SHA,
                module_names=("gears", "gears.model", "gears.utils"),
            )
        )
        from gears.model import GEARS_Model
        from gears.utils import GeneSimNetwork, get_similarity_network, loss_fct

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        seen = np.array(fit["seen_expression_gene_indices"], dtype=np.int64)
        unseen = np.setdiff1d(np.arange(state["gene_emb.weight"].shape[0]), seen)
        result["decoder_parameters"] = {
            key: {
                name: statistics(
                    value.index_select(0 if key.endswith("1") else 1, torch.as_tensor(ids)).numpy()
                )
                for name, ids in (("seen", seen), ("unseen", unseen))
            }
            for key, value in state.items()
            if key in ("indv_w1", "indv_b1", "indv_w2", "indv_b2")
        }
        # Rebuild only frozen graphs, never training pairs, labels or ranking.
        assets = Path(rt["gears_assets"])
        for name, digest in rt["gears_asset_sha256"].items():
            assert sha256_file(assets / name) == digest
        with (assets / "gene2go_all.pkl").open("rb") as f:
            go = pickle.load(f)
        with (assets / "essential_all_data_pert_genes.pkl").open("rb") as f:
            essential = pickle.load(f)
        perts = list(np.unique([g for g in essential if g in go]))
        assert len(perts) == state["pert_emb.weight"].shape[0]
        data = stack.enter_context(
            CanonicalEvaluationData(
                dataset_id=cfg["data"]["dataset_id"],
                protocol_id=cfg["data"]["protocol_id"],
                split_name="test",
                data_root=rt["data_root"],
            )
        )
        reference = load_evaluation_state(
            dataset_id=cfg["data"]["dataset_id"],
            protocol_id=cfg["data"]["protocol_id"],
            data_root=rt["data_root"],
            evaluation_protocol="v2",
        )
        require_evaluation_identity(data, reference, cfg)
        assert tuple(reference.manifest.condition_ids) == tuple(data.split.test_conditions)
        genes = list(data.expression_gene_ids)
        h5 = anndata.read_h5ad(
            run / "fit/official_data/expression_holdout/perturb_processed.h5ad", backed="r"
        )
        assert list(map(str, h5.var.gene_name)) == genes
        h5.file.close()
        graph_dir = run / "fit/official_data"
        co = pd.read_csv(
            graph_dir / "expression_holdout/custom_1_None_0.4_20_co_expression_network.csv"
        )
        assert (
            sha256_file(graph_dir / "go_essential_all/go_essential_all.csv")
            == rt["gears_go_csv_sha256"]
        )
        go_edges = get_similarity_network(
            "go",
            None,
            0.4,
            20,
            str(graph_dir),
            "expression_holdout",
            "custom",
            1,
            None,
            {},
            pert_list=perts,
        )
        co_graph = GeneSimNetwork(co, genes, {g: i for i, g in enumerate(genes)})
        go_graph = GeneSimNetwork(go_edges, perts, {g: i for i, g in enumerate(perts)})
        result["graph_hashes"] = {
            k: sha256_json(v.tolist())
            for k, v in {
                "co_edges": co_graph.edge_index,
                "co_weights": co_graph.edge_weight,
                "go_edges": go_graph.edge_index,
                "go_weights": go_graph.edge_weight,
            }.items()
        }
        params = dict(
            cfg["model"],
            device=args.device,
            num_genes=len(genes),
            num_perts=len(perts),
            G_coexpress=co_graph.edge_index,
            G_coexpress_weight=co_graph.edge_weight,
            G_go=go_graph.edge_index,
            G_go_weight=go_graph.edge_weight,
        )
        net = GEARS_Model(params).to(args.device).eval()
        net.load_state_dict(state, strict=True)
        fitted = FittedAdapter(
            SimpleNamespace(model=net, device=args.device, gene_list=genes, pert_list=perts), {}
        )
        raw = []

        def capture(_module, inp, out):
            w = torch.cat([inp[0].unsqueeze(-1), out[:, None, :].expand(-1, len(genes), -1)], dim=2)
            score = (w * net.indv_w2).sum(dim=2) + net.indv_b2
            score64 = (w.double() * net.indv_w2.double()).sum(dim=2) + net.indv_b2.double()
            raw.append(
                {
                    "float32": statistics(score[:, unseen].detach().cpu().numpy()),
                    "float64_same_features": statistics(score64[:, unseen].detach().cpu().numpy()),
                }
            )

        hook = net.cross_gene_state.register_forward_hook(capture)
        stack.callback(hook.remove)
        rows = []
        for index in (
            0,
            len(data.control_manifest.draws) // 2,
            len(data.control_manifest.draws) - 1,
        ):
            if args.device == "cuda:0":
                admit(policy, snapshot(os.getpid()))
            draw = data.control_manifest.draws[index]
            controls = data.load_control_rows(tuple(draw.ordered_row_ids)).expression
            truth = data.load_truth_rows(draw.condition_id).expression
            targets = [g for g in draw.condition_id.split("+") if g != "ctrl"]
            raw.clear()
            pred = predict(
                fitted, controls, targets, genes, cell_batch=128, query_count=1000, seed=1
            )
            result_delta = pred[:, unseen].astype(np.float64) - controls[:, unseen].astype(
                np.float64
            )
            pred_delta = (
                pred.astype(np.float64).mean(0)[unseen]
                - controls.astype(np.float64).mean(0)[unseen]
            )
            truth_delta = (
                truth.astype(np.float64).mean(0)[unseen]
                - controls.astype(np.float64).mean(0)[unseen]
            )
            args_metric = dict(
                condition_id=draw.condition_id,
                prediction=pred,
                input_control=controls,
                truth=truth,
                metric_control_pool_mean=reference.metric_control_means[index],
                de_gene_indices=reference.manifest.de_gene_indices[draw.condition_id],
                systema_reference=reference.systema_reference,
                de_unavailable_reason=reference.manifest.de_unavailable_reasons.get(
                    draw.condition_id
                ),
                gene_indices=unseen.tolist(),
            )
            computed = compute_condition_metrics_v2(**args_metric)
            all_metrics = compute_condition_metrics_v2(
                **{k: v for k, v in args_metric.items() if k != "gene_indices"}
            )
            manual, reason = pearson_correlation(pred_delta, truth_delta)
            rows.append(
                {
                    "condition": draw.condition_id,
                    "control_count": len(controls),
                    "control_ids_sha256": sha256_json(list(draw.ordered_row_ids)),
                    "unseen_cell_delta": statistics(result_delta),
                    "unseen_mean_delta": statistics(pred_delta),
                    "unseen_truth_delta": statistics(truth_delta),
                    "raw_decoder_delta": list(raw),
                    "manual_txpert": manual,
                    "manual_reason": reason,
                    "metric_txpert": computed.results[0].value,
                    "metric_reason": computed.results[0].reason,
                    "all_metric_replay": [
                        {
                            "metric_id": m.metric_id,
                            "value": m.value,
                            "recorded": terminal["condition_metrics"][index]["metrics"]["results"][
                                i
                            ]["value"],
                        }
                        for i, m in enumerate(all_metrics.results)
                    ],
                    "recorded_reason": terminal["condition_metrics"][index]["gene_groups"][
                        "unseen_expression"
                    ]["results"][0]["reason"],
                }
            )
        hook.remove()
        # Exact official masked loss: excluded final rows receive zero gradient.
        net.zero_grad(set_to_none=True)
        from gears.utils import create_cell_graph_for_prediction
        from torch_geometric.data import Batch

        count = min(2, len(truth))
        batch = Batch.from_data_list(
            [
                create_cell_graph_for_prediction(x, [perts.index(t) for t in targets], targets)
                for x in controls[:count]
            ]
        ).to(args.device)
        prediction = net(batch)
        loss = masked_official_loss(loss_fct, seen)(
            prediction,
            torch.as_tensor(truth[:count], device=args.device),
            [draw.condition_id] * count,
            ctrl=torch.as_tensor(controls[:2].mean(0), device=args.device),
            dict_filter={draw.condition_id: seen.tolist()},
            direction_lambda=0.1,
        )
        loss.backward()
        result["masked_gradient"] = {
            k: statistics(getattr(net, k).grad[:, unseen].detach().cpu().numpy())
            for k in ("indv_w2", "indv_b2")
        }
        result.update(
            conditions=rows,
            checkpoint_unchanged=sha256_file(checkpoint) == before,
            device=args.device,
            wall_seconds=time.monotonic() - started,
        )
        if args.device == "cuda:0":
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "receipt": str(args.output),
                "sha256": sha256_file(args.output),
                "conditions": rows,
                "masked_gradient": result["masked_gradient"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "runtime", "config", "output", "publication"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--publication-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    audit(parser.parse_args())
