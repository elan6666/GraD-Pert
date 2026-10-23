"""Frozen SSL2 CLS encoder/head interventions; aggregate output only.

The capture context surrounds the existing native panel forward. Its exit runs
head-only replay after the inner Collector has closed. No encoder, optimizer,
Teacher EMA, center or temperature setting is changed. Raw rows remain in memory.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
from cap40_posthoc_stats import Distribution, Moments
from grouped_view_diagnostics import GroupedViewBank, feature_statistics
from prototype_pair_diagnostics import PrototypePairBank

_PATHS = {
    "SS": ("student", "student"),
    "ST": ("student", "teacher"),
    "TS": ("teacher", "student"),
    "TT": ("teacher", "teacher"),
}
_FACTORS = (0.5, 1.0, 2.0)


def _array(value):
    return value.detach().float().cpu().numpy().copy()


def _identity(row):
    required = ("context_id", "condition_id", "control_row_id")
    if any(not str(row.get(name, "")) for name in required):
        raise ValueError("Teacher signal records require context/condition/control identities")
    return tuple(str(row.get(name, "")) for name in ("cell_line", "cell_type", *required))


def _maxabs(a, b):
    if a.shape != b.shape:
        raise ValueError("native replay tensor shape differs")
    value = float((a.detach().double() - b.detach().double()).abs().max()) if a.numel() else 0.0
    if not math.isfinite(value):
        raise ValueError("nonfinite native replay")
    return value


def _replay(head, values):
    """Use the real checkpoint head, including historical heads without the helper."""
    # RI predates the shared-forward convenience API. Observe its actual forward,
    # rather than reconstructing a possibly replaced features implementation.
    observed = []
    handle = head.features.register_forward_hook(
        lambda _module, _inputs, output: observed.append(output)
    )
    try:
        logits = head(values)
    finally:
        handle.remove()
    if isinstance(logits, tuple):
        if len(logits) != 2:
            raise ValueError("Projector replay tuple must contain features and logits")
        logits = logits[1]
    if len(observed) != 1:
        raise ValueError("Projector replay requires one features tensor")
    return observed[0], logits, "native_forward_with_scoped_features_hook"


class TeacherSignalBank:
    """Capture Global0/1 only and localize signal across four existing head paths."""

    def __init__(self, tolerance=2e-6):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("finite nonnegative native replay tolerance required")
        self.tolerance = tolerance
        self._entries = []
        self._receipts = []
        self._center = None
        self._temperature = None
        self._active = None
        self._capturing = False
        self._failed = False

    @contextmanager
    def capture(self, objective, records):
        """Hooks are removed even on failure; replay only follows successful capture."""
        import torch

        if self._capturing or self._failed:
            raise ValueError("Teacher signal capture is nested or previously failed")
        active = bool(objective.lambda2 and objective.weights[1][0])
        if self._active is not None and active != self._active:
            raise ValueError("active SSL2 CLS state changed between panel groups")
        self._active = active
        if not active:
            yield self
            return
        if objective.training:
            raise ValueError("Teacher signal capture requires evaluation mode")
        rows = [dict(row) for row in records]
        if not rows:
            raise ValueError("Teacher signal capture requires nonempty records")
        for row in rows:
            _identity(row)
        temperature = float(objective.teacher_temperature)
        center = _array(objective.ssl2_cls_center)
        if (
            not math.isfinite(temperature)
            or temperature <= 0
            or center.ndim != 1
            or not np.isfinite(center).all()
        ):
            raise ValueError("finite frozen SSL2 center and positive temperature required")
        if self._center is not None and (
            temperature != self._temperature or not np.array_equal(center, self._center)
        ):
            raise ValueError("SSL2 center or temperature changed between panel groups")
        self._center, self._temperature = center, temperature
        heads = {role: getattr(objective, role).ssl2_cls for role in ("student", "teacher")}
        counts, features, captured, handles = defaultdict(int), defaultdict(list), {}, []
        self._capturing = True
        try:
            for role, head in heads.items():

                def features_hook(_module, _inputs, output, role=role):
                    if len(features[role]) < 2:
                        features[role].append(output.detach().clone())

                def head_hook(_module, inputs, output, role=role):
                    index = counts[role]
                    counts[role] += 1
                    if index >= 2:
                        return  # Student Local views are not cross-over encoders.
                    logits = output[1] if isinstance(output, tuple) else output
                    if len(features[role]) != index + 1:
                        raise ValueError("SSL2 features/head capture order differs")
                    x, bottleneck = inputs[0], features[role][index]
                    if (
                        x.ndim != 2
                        or bottleneck.ndim != 2
                        or logits.ndim != 2
                        or not len(rows) == len(x) == len(bottleneck) == len(logits)
                        or logits.shape[1] != len(center)
                    ):
                        raise ValueError("SSL2 captured tensors differ from panel records")
                    captured[role, index] = (
                        x.detach().clone(),
                        bottleneck,
                        logits.detach().clone(),
                    )

                handles.append(head.features.register_forward_hook(features_hook))
                handles.append(head.register_forward_hook(head_hook))
            yield self
        except BaseException:
            self._failed = True
            raise
        finally:
            for handle in handles:
                handle.remove()
            self._capturing = False
        try:
            if float(objective.teacher_temperature) != temperature or not np.array_equal(
                _array(objective.ssl2_cls_center), center
            ):
                raise ValueError("SSL2 center or temperature changed during capture")
            if counts["teacher"] != 2 or counts["student"] not in (2, 4):
                raise ValueError("SSL2 requires two Teacher Globals and two/four Student views")
            if len(captured) != 4:
                raise ValueError("SSL2 Global0/1 capture is incomplete")
            # Original Collector must already have exited. Extra hooks would map
            # head replay onto invented native views and corrupt the old bank.
            if any(head._forward_hooks or head.features._forward_hooks for head in heads.values()):
                raise ValueError("native Collector/head hooks must close before Teacher replay")
            parity, methods, staged = [], {}, []
            with torch.no_grad():
                for index in range(2):
                    for path, (encoder_role, head_role) in _PATHS.items():
                        x = captured[encoder_role, index][0]
                        b, logits, method = _replay(heads[head_role], x)
                        methods[path] = method
                        if path in ("SS", "TT"):
                            native = captured[encoder_role, index]
                            b_error, l_error = _maxabs(b, native[1]), _maxabs(logits, native[2])
                            if max(b_error, l_error) > self.tolerance:
                                raise ValueError("native SSL2 replay exceeds tolerance")
                            parity.append(
                                dict(
                                    path=path,
                                    view=f"global{index}",
                                    bottleneck_maxabs=b_error,
                                    logits_maxabs=l_error,
                                    input_dtype=str(x.dtype),
                                    output_dtype=str(logits.dtype),
                                )
                            )
                            # Reuse captured native values after replay parity.
                            b, logits = native[1:]
                        unit = torch.nn.functional.normalize(b.float(), dim=-1)
                        staged.append(
                            dict(
                                path=path,
                                view=f"global{index}",
                                records=rows,
                                values=dict(
                                    input=_array(x),
                                    bottleneck_pre_l2=_array(b),
                                    bottleneck_unit=_array(unit),
                                    raw_logits=_array(logits),
                                ),
                            )
                        )
            self._entries.extend(staged)
            self._receipts.append(
                dict(
                    rows=len(rows),
                    native_head_calls=dict(counts),
                    captured_globals_per_role=2,
                    head_only_replays=8,
                    encoder_recompute=0,
                    parity=parity,
                    replay_methods=methods,
                )
            )
        except BaseException:
            self._failed = True
            raise

    def _unique(self):
        banks = defaultdict(dict)
        duplicates = 0
        for entry in self._entries:
            bank = banks[entry["path"], entry["view"]]
            for i, row in enumerate(entry["records"]):
                key = _identity(row)
                if key in bank:
                    duplicates += 1
                else:
                    bank[key] = (row, {name: value[i] for name, value in entry["values"].items()})
        return banks, duplicates

    def result(self):
        if self._capturing or self._failed:
            raise ValueError("incomplete or failed Teacher signal bank cannot report success")
        base = dict(
            schema="teacher-signal-localization-1",
            updates=0,
            encoder_recompute=0,
            tolerance=self.tolerance,
            captures=self._receipts,
            interpretation="Frozen head interventions; no task-score or model-selection claim",
        )
        if self._active is None:
            raise ValueError("Teacher signal bank has not observed a panel group")
        if not self._active:
            return dict(
                **base,
                status="unavailable:ssl2_cls_disabled",
                paths=[],
                sensitivity=[],
                pairs=[],
                native_pairs=[],
                feature_pairs=[],
            )
        if not self._entries:
            raise ValueError("active SSL2 CLS bank has no observed Globals")
        banks, duplicates = self._unique()
        paths, sensitivity, pairs = [], [], []
        feature_bank = GroupedViewBank()
        arrays = {}
        for (path, view), leaf in sorted(banks.items()):
            rows, data = zip(*leaf.values(), strict=True)
            arrays[path, view] = (
                list(rows),
                {name: np.stack([x[name] for x in data]) for name in data[0]},
            )
            rows, values = arrays[path, view]
            for stage, x in values.items():
                if stage == "raw_logits":
                    stats = feature_statistics(x)  # row Gram, never K-by-K covariance
                else:
                    moments = Moments(x.shape[1])
                    moments.add(x)
                    stats = moments.result()
                ceiling = stats.get("rank_ceiling")
                stats["normalized_rank"] = stats["effective_rank"] / ceiling if ceiling else None
                paths.append(
                    dict(
                        path=path,
                        encoder_role=_PATHS[path][0],
                        head_role=_PATHS[path][1],
                        view=view,
                        stage=stage,
                        aggregation="unique_condition_control_rows_across_contexts",
                        contexts=len({r["context_id"] for r in rows}),
                        conditions=len({r["condition_id"] for r in rows}),
                        **stats,
                    )
                )
                if stage != "raw_logits":
                    feature_bank.add(stage, path, view, rows, x)
        native = PrototypePairBank()
        for view in ("global0", "global1"):
            rows, values = arrays["SS", view]
            native.add("ssl2_cls", "student", view, rows, values["raw_logits"] / 0.1)
            rows, values = arrays["TT", view]
            native.add(
                "ssl2_cls",
                "teacher",
                view,
                rows,
                (values["raw_logits"] - self._center) / self._temperature,
            )
        for path in _PATHS:
            for center_mode in ("frozen_teacher", "zero"):
                center = self._center if center_mode == "frozen_teacher" else 0.0
                for factor in _FACTORS:
                    pair_bank = PrototypePairBank()
                    for view in ("global0", "global1"):
                        rows, values = arrays[path, view]
                        logits = values["raw_logits"].astype(np.float64)
                        scaled = (logits - center) / (self._temperature * factor)
                        dist = Distribution(logits.shape[1])
                        dist.add(scaled)
                        sensitivity.append(
                            dict(
                                path=path,
                                encoder_role=_PATHS[path][0],
                                head_role=_PATHS[path][1],
                                view=view,
                                center_mode=center_mode,
                                temperature_factor=factor,
                                temperature=self._temperature * factor,
                                raw_logit_row_std_mean=float(logits.std(1).mean()),
                                panel_mean_minus_frozen_center_l2=float(
                                    np.linalg.norm(logits.mean(0) - self._center)
                                ),
                                **dist.result(),
                            )
                        )
                        pair_bank.add("ssl2_cls", "teacher", view, rows, scaled)
                        student_rows, student = arrays["SS", view]
                        pair_bank.add(
                            "ssl2_cls", "student", view, student_rows, student["raw_logits"] / 0.1
                        )
                    labels = dict(
                        teacher_path=path,
                        teacher_encoder_role=_PATHS[path][0],
                        teacher_head_role=_PATHS[path][1],
                        center_mode=center_mode,
                        temperature_factor=factor,
                        student_reference="SS_native_temp_0.1",
                    )
                    pairs.extend({**row, **labels} for row in pair_bank.result()["pairs"])
        features = feature_bank.result()
        return dict(
            **base,
            status="completed",
            active_head="ssl2_cls",
            paths=paths,
            sensitivity=sensitivity,
            pairs=pairs,
            native_pairs=native.result()["pairs"],
            feature_pairs=features["pairs"],
            context_features=features["features"],
            duplicate_path_view_rows_ignored=duplicates,
            teacher_temperature=self._temperature,
            center_l2=float(np.linalg.norm(self._center)),
            limits=[
                "Cross-over feature bases may rotate; swaps alone are not causal proof.",
                "Only Global0/1 encoders captured; no synthetic Teacher Local output.",
                "Sensitivity uses common frozen Teacher center or zero; not new training.",
                "Sensitivity Teacher paths pair against original SS at Student temperature 0.1.",
                "Prototype pairing is context matched and ordered condition-pair equal.",
                "Raw rows retained only in memory; results contain scalar aggregates.",
            ],
        )
