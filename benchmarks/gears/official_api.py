"""Narrow calls to the frozen official GEARS API; no local model implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GearsOfficialModules:
    package: ModuleType
    utils: ModuleType
    data_utils: ModuleType
    torch: ModuleType
    pyg_loader: ModuleType

    @classmethod
    def from_mapping(cls, modules: Mapping[str, ModuleType]) -> GearsOfficialModules:
        return cls(
            package=modules["gears"],
            utils=modules["gears.utils"],
            data_utils=modules["gears.data_utils"],
            torch=modules["torch"],
            pyg_loader=modules["torch_geometric.loader"],
        )


@dataclass(frozen=True)
class GearsModelParameters:
    hidden_size: int
    num_go_gnn_layers: int
    num_gene_gnn_layers: int
    decoder_hidden_size: int
    num_similar_genes_go_graph: int
    num_similar_genes_co_express_graph: int
    coexpress_threshold: float
    uncertainty: bool
    uncertainty_reg: float
    direction_lambda: float
    no_perturb: bool

    def official_kwargs(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "num_go_gnn_layers": self.num_go_gnn_layers,
            "num_gene_gnn_layers": self.num_gene_gnn_layers,
            "decoder_hidden_size": self.decoder_hidden_size,
            "num_similar_genes_go_graph": self.num_similar_genes_go_graph,
            "num_similar_genes_co_express_graph": self.num_similar_genes_co_express_graph,
            "coexpress_threshold": self.coexpress_threshold,
            "uncertainty": self.uncertainty,
            "uncertainty_reg": self.uncertainty_reg,
            "direction_lambda": self.direction_lambda,
            "no_perturb": self.no_perturb,
        }


class OfficialGearsAPI:
    """Adapter that delegates construction, training, and forward to GEARS."""

    def __init__(self, modules: GearsOfficialModules) -> None:
        self.modules = modules
        for symbol in ("PertData", "GEARS"):
            if not hasattr(modules.package, symbol):
                raise ValueError(f"official gears package lacks {symbol}")
        for symbol in ("create_cell_graph_for_prediction",):
            if not hasattr(modules.utils, symbol):
                raise ValueError(f"official gears.utils lacks {symbol}")
        for symbol in (
            "get_DE_genes",
            "rank_genes_groups_by_cov",
            "get_dropout_non_zero_genes",
        ):
            if not hasattr(modules.data_utils, symbol):
                raise ValueError(f"official gears.data_utils lacks {symbol}")

    def _materialize_training_de_metadata(self, adata: Any) -> tuple[Any, dict[str, object]]:
        """Build official DE rankings without dropping singleton conditions."""

        condition_names = adata.obs["condition_name"].astype(str)
        conditions = adata.obs["condition"].astype(str)
        counts = condition_names.value_counts()
        singleton_names = tuple(
            sorted(str(name) for name, count in counts.items() if int(count) < 2)
        )
        control_names = tuple(sorted(set(condition_names[conditions == "ctrl"].tolist())))
        if len(control_names) != 1:
            raise ValueError(f"official GEARS control DE context is ambiguous: {control_names}")
        if control_names[0] in singleton_names:
            raise ValueError("official GEARS DE ranking requires at least two control cells")

        rankable_mask = ~condition_names.isin(singleton_names)
        rankable_adata = adata[np.asarray(rankable_mask, dtype=bool)].copy()
        rankable_adata.obs["condition_name"] = (
            rankable_adata.obs["condition_name"].astype(str).astype("category")
        )
        rankings_raw = self.modules.data_utils.rank_genes_groups_by_cov(
            rankable_adata,
            groupby="condition_name",
            covariate="cell_type",
            control_group="ctrl_1",
            n_genes=int(adata.n_vars),
            key_added="rank_genes_groups_cov_all",
            return_dict=True,
        )
        if not isinstance(rankings_raw, Mapping):
            raise TypeError("official GEARS DE ranking did not return a mapping")
        rankings = {
            str(name): [str(gene_id) for gene_id in values] for name, values in rankings_raw.items()
        }

        noncontrol_names = set(condition_names[conditions != "ctrl"].tolist())
        fallback_names = tuple(sorted(set(singleton_names) & noncontrol_names))
        expression_gene_ids = [str(gene_id) for gene_id in adata.var_names]
        if len(expression_gene_ids) < 2:
            raise ValueError("official GEARS DE fallback requires at least two genes")
        for name in fallback_names:
            # The common evaluator already marks singleton-condition DE metrics
            # unavailable. This stable full-axis order is used only to keep the
            # frozen official one-epoch train/validation bookkeeping defined.
            rankings[name] = list(expression_gene_ids)

        missing = sorted(noncontrol_names - set(rankings))
        if missing:
            raise ValueError(f"official GEARS DE rankings lack conditions: {missing}")
        adata.uns["rank_genes_groups_cov_all"] = {
            name: np.asarray(gene_ids, dtype=str) for name, gene_ids in rankings.items()
        }
        adata = self.modules.data_utils.get_dropout_non_zero_genes(adata)
        receipt = {
            "schema_version": "official-gears-de-ranking-v1",
            "truth_scope": "train_validation_only",
            "official_ranked_condition_count": len(noncontrol_names) - len(fallback_names),
            "singleton_fallback_condition_count": len(fallback_names),
            "singleton_fallback_condition_names": list(fallback_names),
            "singleton_fallback_policy": "stable_full_gene_axis_internal_metrics_only",
        }
        return adata, receipt

    @staticmethod
    def _validate_training_graph_cache(pert_data: Any) -> dict[str, int | bool]:
        """Validate official PyG graphs built after DE metadata was attached."""

        dataset_processed = pert_data.dataset_processed
        if not isinstance(dataset_processed, Mapping) or not dataset_processed:
            raise ValueError("official GEARS graph cache contains no condition graphs")

        expected_de_index_count = 20
        verified_condition_count = 0
        for condition, graphs in dataset_processed.items():
            if not isinstance(graphs, Sequence) or not graphs:
                raise ValueError(f"official GEARS graph cache is empty: {condition}")
            de_idx = np.asarray(graphs[0].de_idx).reshape(-1)
            if de_idx.size != expected_de_index_count:
                raise ValueError(
                    "official GEARS graph cache has invalid DE index count: "
                    f"{condition}={de_idx.size}"
                )
            verified_condition_count += 1

        return {
            "graph_cache_built_after_de_ranking": True,
            "graph_de_index_count": expected_de_index_count,
            "graph_condition_count": verified_condition_count,
        }

    def prepare_training_data(
        self,
        *,
        data_root: str | Path,
        dataset_id: str,
        training_validation_adata: Any,
        split_pickle_path: str | Path,
        train_batch_size: int,
        eval_batch_size: int,
    ) -> Any:
        """Call official data APIs on data containing no test perturbation rows."""

        pert_data = self.modules.package.PertData(str(Path(data_root)))
        prepared_adata = self.modules.data_utils.get_DE_genes(
            training_validation_adata,
            skip_calc_de=True,
        )
        prepared_adata, receipt = self._materialize_training_de_metadata(prepared_adata)
        pert_data.new_data_process(
            dataset_id,
            adata=prepared_adata,
            # Official condition names and safe ranking metadata already exist,
            # so the frozen graph builder sees them on its only construction.
            skip_calc_de=True,
        )
        receipt.update(self._validate_training_graph_cache(pert_data))
        pert_data.gradpert_de_ranking_receipt = receipt
        non_zeros: dict[str, np.ndarray[Any, Any]] = {}
        for condition in pert_data.adata.obs["condition"].astype(str).unique():
            subset = pert_data.adata[pert_data.adata.obs["condition"].astype(str) == condition]
            mean = np.asarray(subset.X.mean(axis=0)).reshape(-1)
            names = subset.obs["condition_name"].astype(str).unique()
            if len(names) != 1:
                raise ValueError(f"official GEARS condition name is ambiguous: {condition}")
            non_zeros[str(names[0])] = np.sort(np.flatnonzero(mean != 0))
        pert_data.adata.uns["non_zeros_gene_idx"] = non_zeros
        pert_data.adata.write_h5ad(Path(pert_data.dataset_path) / "perturb_processed.h5ad")
        pert_data.prepare_split(
            split="custom",
            seed=1,
            split_dict_path=str(Path(split_pickle_path)),
        )
        pert_data.get_dataloader(
            batch_size=int(train_batch_size),
            test_batch_size=int(eval_batch_size),
        )
        # The prediction phase is separate and truth-free. The official train
        # routine therefore receives no test loader and cannot evaluate test.
        pert_data.dataloader.pop("test_loader", None)
        return pert_data

    @staticmethod
    def require_perturbation_coverage(
        pert_data: Any,
        condition_ids: Sequence[str],
    ) -> tuple[str, ...]:
        available = set(np.asarray(pert_data.pert_names, dtype=str).tolist())
        targets = tuple(
            sorted(
                {
                    component
                    for condition in condition_ids
                    for component in condition.split("+")
                    if component != "ctrl"
                }
            )
        )
        missing = sorted(set(targets) - available)
        if missing:
            raise ValueError(f"official GEARS perturbation graph lacks targets: {missing}")
        return targets

    def fit_one_epoch(
        self,
        *,
        pert_data: Any,
        parameters: GearsModelParameters,
        learning_rate: float,
        weight_decay: float,
        checkpoint_dir: str | Path,
        device: str,
        experiment_name: str,
    ) -> Any:
        model = self.modules.package.GEARS(
            pert_data,
            device=device,
            weight_bias_track=False,
            proj_name="GraD-Pert-benchmark",
            exp_name=experiment_name,
        )
        model.model_initialize(**parameters.official_kwargs())
        model.train(
            epochs=1,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        checkpoint = Path(checkpoint_dir)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(checkpoint))
        return model

    def predict_exact_controls(
        self,
        *,
        trained_model: Any,
        perturbation_genes: Sequence[str],
        input_controls: np.ndarray[Any, Any],
        batch_size: int,
    ) -> np.ndarray[Any, Any]:
        """Run official graph construction/model forward for the exact 300 rows."""

        controls = np.asarray(input_controls, dtype=np.float32)
        if controls.ndim != 2 or controls.shape[0] != 300:
            raise ValueError("official GEARS prediction requires exact [300,G] controls")
        pert_names = np.asarray(trained_model.pert_list, dtype=str)
        pert_idx: list[int] = []
        for gene in perturbation_genes:
            matches = np.flatnonzero(pert_names == str(gene))
            if matches.size != 1:
                raise ValueError(
                    f"perturbation is absent/ambiguous in official GEARS graph: {gene}"
                )
            pert_idx.append(int(matches[0]))
        graphs = [
            self.modules.utils.create_cell_graph_for_prediction(
                row,
                pert_idx,
                list(perturbation_genes),
            )
            for row in controls
        ]
        loader = self.modules.pyg_loader.DataLoader(
            graphs,
            batch_size=int(batch_size),
            shuffle=False,
        )
        official_model = trained_model.best_model.to(trained_model.device)
        official_model.eval()
        outputs: list[np.ndarray[Any, Any]] = []
        with self.modules.torch.no_grad():
            for batch in loader:
                batch = batch.to(trained_model.device)
                prediction = official_model(batch)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                outputs.append(prediction.detach().cpu().numpy())
        if not outputs:
            raise ValueError("official GEARS forward produced no predictions")
        result = np.asarray(np.vstack(outputs), dtype=np.float32)
        if result.shape != controls.shape or not np.isfinite(result).all():
            raise ValueError(f"official GEARS prediction shape/finite gate failed: {result.shape}")
        return result
