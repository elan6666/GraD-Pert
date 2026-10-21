"""Read-only Response hooks for a fixed, identity-aligned diagnostic panel.

The caller owns sampling, eval/no_grad mode, and checkpoint/RNG verification.
Only scalar aggregate rows survive a hook; complete activations are never kept.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from cap40_posthoc_accel import sample_moments
from cap40_posthoc_stats import digest_json, seeded_int
from torch import Tensor, nn


def spectrum(values: np.ndarray) -> dict[str, Any]:
    """Existing medium covariance definition, with explicit common-mean energy."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty row-by-feature matrix required")
    result = sample_moments(x)
    mean_energy = float(np.square(x.mean(0)).sum())
    result["mean_energy"] = mean_energy
    result["absolute_variance"] = float(x.var(0).sum())
    result["total_energy"] = float(np.square(x).sum() / len(x))
    ceiling = result.get("rank_ceiling", 0)
    result["normalized_effective_rank"] = result["effective_rank"] / ceiling if ceiling else None
    return result


class ResponseStageObserver:
    """Attach to ``obj.student.response`` around exactly one diagnostic forward.

    ``gene_ids`` are the exact ordered query IDs, excluding the appended CLS.
    ``records`` retain condition_id, context_id and control_row_id for every row.
    Use a single fixed panel for all condition/control interventions. No model
    parameters, modes, outputs, gradients or random streams are changed here.
    ``injection_branch`` is the MLP result; ``injection`` is the actual state
    entering self-attention, including any injection residual (or bypass).
    """

    def __init__(
        self,
        response: nn.Module,
        *,
        gene_ids: Sequence[int] | Tensor,
        records: Sequence[Mapping[str, Any]],
        seed: int = 1,
        max_genes: int = 128,
    ) -> None:
        if max_genes < 2:
            raise ValueError("at least two sampled genes required")
        ids = gene_ids.detach().cpu().tolist() if isinstance(gene_ids, Tensor) else list(gene_ids)
        self.ids = [int(g) for g in ids]
        if len(self.ids) < 2 or len(set(self.ids)) != len(self.ids):
            raise ValueError("unique query gene IDs required")
        self.records = [dict(r) for r in records]
        required = ("condition_id", "context_id", "control_row_id")
        if not self.records or any(any(k not in r for k in required) for r in self.records):
            raise ValueError("records need condition, context and control identities")
        for name in ("injections", "self_layers", "cross_layers", "ffn_layers", "norm"):
            if not hasattr(response, name):
                raise ValueError("RelayResponseEncoder hook target required: " + name)
        self.response = response
        self.seed = seed
        self.selected = sorted(
            range(len(ids)), key=lambda i: (seeded_int(self.ids[i], seed), self.ids[i])
        )[:max_genes]
        self._handles: list[Any] = []
        self._active = False
        self._entered = False
        self._calls = 0
        self._failed = False
        self._rows: dict[str, list[dict[str, Any]]] = {
            "cell_statistics": [],
            "group_statistics": [],
            "pair_statistics": [],
        }

    def __enter__(self) -> ResponseStageObserver:
        if self._entered:
            raise ValueError("observer is single-use")
        self._entered = self._active = True
        try:
            self._handles.append(self.response.register_forward_pre_hook(self._entry))
            for index, module in enumerate(self.response.injections):
                prefix = f"layer{index + 1}"
                self._handles.append(
                    module.register_forward_pre_hook(
                        lambda m, args, p=prefix: self._observe(
                            p + ".injection_input", args[0][..., : args[0].shape[-1] // 2]
                        )
                    )
                )
                self._handles.append(
                    module.register_forward_hook(
                        lambda m, args, output, p=prefix: self._observe(
                            p + ".injection_branch", output
                        )
                    )
                )
            for name in ("self", "cross", "ffn"):
                for index, module in enumerate(getattr(self.response, name + "_layers")):
                    stage = f"layer{index + 1}.{name}"
                    if name == "self":
                        # The MLP hook sees F(x,p), not x+F(x,p) for residual
                        # injection. Observe the actual state at self's input.
                        actual_stage = f"layer{index + 1}.injection"
                        self._handles.append(
                            module.register_forward_pre_hook(
                                lambda m, args, s=actual_stage: self._observe(s, args[0])
                            )
                        )
                    self._handles.append(
                        module.register_forward_hook(
                            lambda m, args, output, s=stage: self._observe(s, output)
                        )
                    )
            self._handles.append(
                self.response.norm.register_forward_pre_hook(
                    lambda m, args: self._observe("final.pre_norm", args[0])
                )
            )
            self._handles.append(
                self.response.norm.register_forward_hook(
                    lambda m, args, output: self._observe("final.post_norm", output)
                )
            )
        except BaseException:
            self.close()
            raise
        return self

    def _entry(self, module: nn.Module, args: tuple[Any, ...]) -> None:
        self._calls += 1
        if self._calls != 1:
            raise ValueError("one forward per observer; create a fresh observer for each panel")
        self._observe("entry", args[0])

    def _observe(self, stage: str, tensor: Tensor) -> None:
        if tensor.ndim not in (3, 4) or tensor.shape[:2] != (len(self.records), len(self.ids) + 1):
            raise ValueError("Response rows/genes/terminal CLS do not match diagnostic identities")
        if not bool(torch.isfinite(tensor.detach()).all()):
            raise ValueError("nonfinite Response activation at " + stage)
        # Only the bounded panel and CLS cross to CPU; no full activation copy.
        index = torch.tensor([*self.selected, len(self.ids)], device=tensor.device)
        x = tensor.detach().index_select(1, index).double().cpu().numpy()
        if x.ndim == 3:
            self._record(stage, "single", x)
        else:
            for stream in range(x.shape[2]):
                self._record(stage, f"stream{stream}", x[:, :, stream])
            self._record(stage, "stream_mean", x.mean(2))

    def _record(self, stage: str, stream: str, x: np.ndarray) -> None:
        base = dict(stage=stage, stream=stream)
        for index, record in enumerate(self.records):
            self._rows["cell_statistics"].append(
                {
                    **base,
                    **record,
                    "statistical_axis": "genes_within_cell",
                    **spectrum(x[index, :-1]),
                }
            )
        contexts: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            contexts[
                tuple(str(r.get(k, "")) for k in ("cell_line", "cell_type", "context_id"))
            ].append(i)
        for context, indices in contexts.items():
            # One realization per (condition,control); duplicated truth rows are not independent.
            unique = {}
            for i in indices:
                unique.setdefault(
                    (str(self.records[i]["condition_id"]), str(self.records[i]["control_row_id"])),
                    i,
                )
            indices = list(unique.values())
            context_meta = dict(zip(("cell_line", "cell_type", "context_id"), context, strict=True))
            conditions: dict[str, list[int]] = defaultdict(list)
            for i in indices:
                conditions[str(self.records[i]["condition_id"])].append(i)
            for condition, group in conditions.items():
                common = {
                    **base,
                    **context_meta,
                    "condition_id": condition,
                    "independent_controls": len(group),
                }
                self._rows["group_statistics"].append(
                    {**common, "statistical_axis": "cls_across_controls", **spectrum(x[group, -1])}
                )
                stats = [spectrum(x[group, g]) for g in range(x.shape[1] - 1)]
                valid = [s for s in stats if s["effective_rank"] is not None]
                self._rows["group_statistics"].append(
                    {
                        **common,
                        "statistical_axis": "same_gene_across_controls_gene_mean",
                        "genes": len(stats),
                        "valid_genes": len(valid),
                        "effective_rank": float(np.mean([s["effective_rank"] for s in valid]))
                        if valid
                        else None,
                        "relative_variance": float(np.mean([s["relative_variance"] for s in valid]))
                        if valid
                        else None,
                        "absolute_variance": float(
                            np.mean([s["absolute_variance"] for s in stats])
                        ),
                        "mean_energy": float(np.mean([s["mean_energy"] for s in stats])),
                        "rank_ceiling": min(x.shape[-1], len(group) - 1),
                        "status": "ok" if valid else "undefined:fewer_than_two_rows",
                    }
                )
            pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for a, i in enumerate(indices):
                for j in indices[a + 1 :]:
                    same_p = self.records[i]["condition_id"] == self.records[j]["condition_id"]
                    same_c = self.records[i]["control_row_id"] == self.records[j]["control_row_id"]
                    label = (
                        "same_condition_different_control"
                        if same_p
                        else "different_condition_same_control"
                        if same_c
                        else "different_condition_different_control"
                    )
                    pairs[label].append((i, j))
            for stratum, pairs_ in pairs.items():
                i, j = np.asarray(pairs_).T
                for name, left, right in (
                    ("cls", x[i, -1], x[j, -1]),
                    ("aligned_gene_token", x[i, :-1], x[j, :-1]),
                ):
                    product = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
                    valid = product > 0
                    cos = (left * right).sum(-1)[valid] / product[valid]
                    self._rows["pair_statistics"].append(
                        {
                            **base,
                            **context_meta,
                            "stratum": stratum,
                            "object": name,
                            "pair_count": len(pairs_),
                            "valid_vector_pairs": int(valid.sum()),
                            "cosine": float(cos.mean()) if len(cos) else None,
                            "mean_squared_distance": float(np.square(left - right).sum(-1).mean()),
                        }
                    )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._failed = exc_type is not None
        self.close()

    def result(self) -> dict[str, Any]:
        if self._active or self._calls != 1:
            raise ValueError("read results after one completed context-managed forward")
        if self._failed or not any(
            r["stage"] == "final.post_norm" for r in self._rows["cell_statistics"]
        ):
            raise ValueError("incomplete Response forward cannot be a diagnostic result")
        return {
            "schema": "response-stage-fixed-panel-1",
            "query_gene_ids_sha256": digest_json(self.ids),
            "sampled_gene_ids": [self.ids[i] for i in self.selected],
            "sampled_gene_ids_sha256": digest_json([self.ids[i] for i in self.selected]),
            "seed": self.seed,
            "rows": len(self.records),
            "scope": "single_forward_fixed_panel; terminal CLS excluded from gene statistics",
            "activation_retention": "none; scalar summaries only",
            **self._rows,
        }
