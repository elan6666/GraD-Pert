"""Read-only head-distribution diagnostics on sealed training inputs/checkpoints.

Execute with PYTHONPATH pointing at the original, immutable training checkout.
The diagnostic script's SHA is recorded separately from the method source SHA.
Only small aggregate statistics are exported; no logits or cell matrices persist.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HEADS = ("ssl1_cls", "ssl1_node", "ssl2_cls", "ssl2_node")


def entropy(p):
    return -(p * p.clamp_min(1e-300).log()).sum(-1)


class Distribution:
    def __init__(self, width):
        self.width = width
        self.count = 0
        self.sum = torch.zeros(width, dtype=torch.float64)
        self.hits = torch.zeros(width, dtype=torch.long)
        self.entropy_sum = self.max_sum = self.logit_std_sum = 0.0

    def add(self, logits):
        x = logits.detach().double().cpu().reshape(-1, self.width)
        p = x.softmax(-1)
        self.count += len(x)
        self.sum += p.sum(0)
        self.hits += torch.bincount(p.argmax(-1), minlength=self.width)
        self.entropy_sum += float(entropy(p).sum())
        self.max_sum += float(p.max(-1).values.sum())
        self.logit_std_sum += float(x.std(-1, unbiased=False).sum())

    def result(self):
        if not self.count:
            return {"samples": 0}
        mean = self.sum / self.count
        h = self.entropy_sum / self.count
        hm = float(entropy(mean))
        values, indices = mean.topk(min(10, self.width))
        return {
            "samples": self.count,
            "mean_sample_entropy": h,
            "entropy_fraction_of_uniform": h / math.log(self.width),
            "marginal_entropy": hm,
            "effective_marginal_prototypes": math.exp(hm),
            "sample_dependence_js": max(0.0, hm - h),
            "mean_top1_probability": self.max_sum / self.count,
            "argmax_prototypes_used": int((self.hits > 0).sum()),
            "largest_argmax_fraction": float(self.hits.max()) / self.count,
            "mean_temperature_scaled_logit_std": self.logit_std_sum / self.count,
            "marginal_top10": list(zip(indices.tolist(), values.tolist(), strict=True)),
        }


class Features:
    def __init__(self):
        self.count = 0
        self.sum = self.outer = self.unit_sum = None
        self.norm_sum = 0.0

    def add(self, features):
        x = features.detach().double().cpu().reshape(-1, features.shape[-1])
        if self.sum is None:
            self.sum = torch.zeros(x.shape[-1], dtype=torch.float64)
            self.outer = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float64)
            self.unit_sum = torch.zeros_like(self.sum)
        self.count += len(x)
        self.sum += x.sum(0)
        self.outer += x.T @ x
        norms = x.norm(dim=-1)
        self.norm_sum += float(norms.sum())
        self.unit_sum += (x / norms.clamp_min(1e-30)[:, None]).sum(0)

    def result(self):
        if not self.count:
            return {"samples": 0}
        mean = self.sum / self.count
        cov = self.outer / self.count - mean[:, None] * mean[None, :]
        eig = torch.linalg.eigvalsh(cov).clamp_min(0)
        trace = float(eig.sum())
        n = self.count
        return {
            "samples": n,
            "mean_norm": self.norm_sum / n,
            "total_centered_variance": trace,
            "relative_centered_variance": trace / max(float(self.outer.trace()) / n, 1e-30),
            "covariance_effective_rank": math.exp(float(entropy(eig / trace))) if trace else 0,
            "mean_pair_cosine": (float(self.unit_sum.square().sum()) - n) / (n * (n - 1))
            if n > 1
            else None,
        }


def run(args):
    from gradpert.config import load_experiment_config
    from gradpert.data._io import atomic_json
    from gradpert.hashing import sha256_file, sha256_json
    from gradpert.training.v2 import objective as module
    from gradpert.training.v2.checkpoint import load_evaluation_checkpoint
    from gradpert.training.v2.runtime import prepare_runtime
    from gradpert.training.v2.views import assemble_batch

    if args.output.exists() or not args.output.resolve().is_relative_to("/data/yilangliu"):
        raise ValueError("diagnostics require a fresh server output")
    source = Path(module.__file__).resolve().parents[4]
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        raise ValueError("method source must be immutable and clean")
    identity = json.loads((args.run_root / "run_manifest.json").read_text())
    if sha != identity["source"]["commit"]:
        raise ValueError("diagnostic method source must equal training SHA")
    config = load_experiment_config(args.config)
    if sha256_json(config.model_dump(mode="json")) != identity["resolved_config_sha256"]:
        raise ValueError("diagnostic config differs from checkpoint")
    journal = json.loads((args.run_root / "fit/epoch_state.json").read_text())
    selected = journal["last"]
    checkpoint = args.run_root / "fit" / selected["file"]
    before_hash = sha256_file(checkpoint)
    if before_hash != selected["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    args.output.mkdir(parents=True)
    handles = []
    distributions = {}
    features = {}
    pairs = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    logits_names = {}
    original_ce = module.cross_entropy_terms
    inputs = []
    losses = []
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.9, device)
    with prepare_runtime(
        config, data_root=args.data_root, run_seed=1, device=device, purpose="evaluation"
    ) as runtime:
        obj = runtime.objective
        if runtime.identity != identity["data"]:
            raise ValueError("diagnostic data identity differs from training")
        progress = load_evaluation_checkpoint(
            checkpoint, obj, training_identity=identity, checkpoint_sha256=before_hash
        )
        obj.eval()
        from capacity_probe import state_fingerprint

        state_before = state_fingerprint(obj.state_dict())
        center_before = {head: getattr(obj, head + "_center").clone() for head in HEADS}
        for role in ("student", "teacher"):
            model = getattr(obj, role)
            for name in HEADS:
                key = role + "." + name
                center = getattr(obj, name + "_center")
                distributions[key] = Distribution(model.options.prototypes)
                features[key + ".input"] = Features()
                features[key + ".bottleneck"] = Features()

                def head_hook(head, values, output, key=key, role=role, center=center):
                    logits_names[id(output)] = key
                    scaled = (
                        output.float() / 0.1
                        if role == "student"
                        else (output.float() - center) / 0.04
                    )
                    distributions[key].add(scaled)
                    features[key + ".input"].add(values[0])

                def feature_hook(head, values, output, key=key):
                    features[key + ".bottleneck"].add(output)

                handles.append(getattr(model, name).register_forward_hook(head_hook))
                handles.append(getattr(model, name).features.register_forward_hook(feature_hook))

        def observed_ce(student, teacher, center):
            value = original_ce(student, teacher, center)
            p = ((teacher.detach().double() - center.double()) / 0.04).softmax(-1)
            h = entropy(p)
            ce = -(p * (student.detach().double() / 0.1).log_softmax(-1)).sum(-1)
            key = logits_names[id(student)]
            tally = pairs[key]
            tally[0] += len(ce)
            tally[1] += float(ce.sum())
            tally[2] += float(h.sum())
            tally[3] += float((ce - h).sum())
            return value

        module.cross_entropy_terms = observed_ce
        try:
            rng = np.random.default_rng(args.seed)
            torch.manual_seed(args.seed)
            raw_stream = runtime.data.iter_train_epoch(
                epoch=0, device=device, batch_size=args.cells, max_unique_conditions=0
            )
            for raw in itertools.islice(raw_stream, args.batches):
                from shared_gpu import admit, snapshot

                admit(args.policy, snapshot(os.getpid()))
                batch = assemble_batch(
                    raw,
                    runtime.index,
                    runtime.options,
                    rng,
                    allowed_expression_ids=runtime.allowed_expression_ids,
                )
                inputs.append(
                    {
                        "rows": sha256_json(list(raw.perturbed_row_ids)),
                        "controls": sha256_json(list(raw.control_row_ids)),
                        "queries": sha256_json(batch.query_positions.cpu().tolist()),
                        "graph_views": sha256_json(
                            [v.ids.cpu().tolist() for v in batch.graph_views]
                        ),
                        "cell_views": sha256_json(
                            [
                                {
                                    "positions": v.positions.cpu().tolist(),
                                    "mask": v.mask.cpu().tolist(),
                                }
                                for v in batch.cell_views
                            ]
                        ),
                    }
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    loss, terms = obj(batch)
                losses.append({"joint": float(loss), **{k: float(v) for k, v in terms.items()}})
                obj.pending.clear()
                logits_names.clear()
                print(
                    json.dumps({"batch_completed": len(inputs), "joint": float(loss)}), flush=True
                )
            raw_stream.close()
        finally:
            module.cross_entropy_terms = original_ce
            for handle in handles:
                handle.remove()
        if len(inputs) != args.batches or obj.pending:
            raise ValueError("incomplete diagnostic input budget")
        if any(not torch.equal(center_before[h], getattr(obj, h + "_center")) for h in HEADS):
            raise ValueError("diagnostic changed teacher centers")
        if sha256_file(checkpoint) != before_hash:
            raise ValueError("diagnostic changed checkpoint")
        if state_before != state_fingerprint(obj.state_dict()):
            raise ValueError("diagnostic changed model/teacher/center state")
        atomic_json(
            args.output / "receipt.json",
            {
                "schema": "prototype-diagnostics-1",
                "arm": args.arm,
                "method_source_sha": sha,
                "diagnostic_script_sha256": sha256_file(Path(__file__)),
                "config_sha256": sha256_file(args.config),
                "checkpoint_sha256": before_hash,
                "checkpoint_epoch": selected["epoch"],
                "history_epochs": len(progress["history"]),
                "data_identity": runtime.identity,
                "split": "train",
                "mode": "eval_bf16_no_updates",
                "cells_per_batch": args.cells,
                "batches": args.batches,
                "view_seed": args.seed,
                "ordered_inputs": inputs,
                "input_sha256": sha256_json(inputs),
                "distributions": {k: v.result() for k, v in distributions.items()},
                "features": {k: v.result() for k, v in features.items()},
                "centers": {
                    h: {
                        "l2": float(getattr(obj, h + "_center").norm()),
                        "prototype_std": float(getattr(obj, h + "_center").std()),
                    }
                    for h in HEADS
                },
                "pairs": {
                    k: {
                        "sample_pair_count": v[0],
                        "cross_entropy": v[1] / v[0],
                        "teacher_entropy": v[2] / v[0],
                        "teacher_to_student_kl": v[3] / v[0],
                    }
                    for k, v in pairs.items()
                },
                "losses": losses,
                "state": {
                    "checkpoint_unchanged": True,
                    "centers_unchanged": True,
                    "optimizer_updates": 0,
                    "teacher_updates": 0,
                    "selection_use": "none",
                },
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run-root", "config", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--resource-queue", type=Path, required=True)
    args = parser.parse_args()
    from run_group import lock
    from shared_gpu import admit, snapshot, validate_policy

    args.policy = json.loads(args.resource_queue.read_text())["shared_gpu_policy"]
    validate_policy(args.policy)
    with lock(Path("/data/yilangliu/GraD-Pert/runtime/v2-gpu-0.lock")):
        admit(args.policy, snapshot(os.getpid()))
        run(args)
