"""Read-only server preflight: resolve sealed data, full model and optimizer.

No training step, scientific result or capacity claim is produced. This catches
real artifact/environment incompatibility before launching GPU fitting.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gpu", choices=("0", "1"), required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    from gradpert.config import load_experiment_config
    from gradpert.execution.identity import inspect_environment, inspect_source_identity
    from gradpert.training.v2.runtime import prepare_runtime

    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(args.config)
    source = inspect_source_identity(
        root, formal=True, expected_repository=config.source_code.repository
    )
    environment = inspect_environment(root, device_name="cuda:0")
    free, total = torch.cuda.mem_get_info()
    if total - free > 512 * 1024**2:
        raise RuntimeError("preflight requires an idle GPU; existing work will not be displaced")
    with prepare_runtime(
        config,
        data_root=args.data_root,
        run_seed=config.training.run_seeds[0],
        device=torch.device("cuda:0"),
    ) as runtime:
        print(
            json.dumps(
                {
                    "status": "constructed_not_trained",
                    "source": source.payload(),
                    "environment": environment.payload(),
                    "data": runtime.identity,
                    "student_parameters": sum(
                        p.numel() for p in runtime.objective.student.parameters()
                    ),
                    "steps_per_epoch": runtime.steps_per_epoch,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
