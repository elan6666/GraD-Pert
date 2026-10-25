"""Strict version-two model parameters without importing tensor runtimes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class V2Architecture:
    width: int = 256
    heads: int = 4
    graph_layers: int = 2
    graph_read_mode: str = "static"
    latent_rank: int = 64
    streams: int = 4
    sinkhorn_backend: str = "native"
    dropout: float = 0.1
    attention: str = "hybrid"
    ffn_type: str = "gelu"
    sparse_topk: int = 500
    sparse_index_dim: int = 64
    sparse_query_chunk: int = 8
    kda_layers: int = 3
    checkpoint_layers: bool = True
    projector_hidden: int = 2048
    projector_bottleneck: int = 256
    prototypes: int = 16384
    relay_eval_seed: int | None = None
    relay_passes: int = 2
    relay_kernel: str = "eager"
    weighted_gram_backend: str = "eager"
    relay_validate_once: bool = False
    short_graph_kernel: bool = False
    cache_kda_constants: bool = False
    relay_scan_chunk_size: int = 32
    relay_graph_chunk_rows: int = 64
    relay_sequence_chunk_size: int | None = None
    graph_source_key_gate: bool = False
    graph_mhc: bool = False
    attention_replacement: str = "none"
    self_readout: str = "final"
    all_kda_prefix: bool = False
    fixed_gene_order: bool = False
    prior_shared_adapter: bool = False
    gene_conditioned_readout: bool = False
    direct_target_flag: bool = False
    learned_genept_projection: bool = False
    checkpoint_genept_mapping: bool = False
    genept_projection_activation: str = "none"
    mlp_profile: str = "historical"
    ffn_variant: str = "mixed"
    ffn_input_norm: str = "historical"
    ssl2_cls_input_layernorm: bool = False
    ffn_clip: bool = True
    ffn_linear_bias: str = "historical"
    prototype_weight_norm: bool = False
    prototype_freeze_epochs: int = 0
    prototype_fixed_scale: bool = False
    prototype_freeze_gradients: bool = False
    prediction_parameterization: str = "residual"
    genept_hidden_width: int = 0
    genept_middle_width: int = 0
    control_conditioned_graph: bool = False
    perturbation_injection: str = "per_layer"
    response_injection_residual: bool = False
    random_gene_order: bool = True

    def __post_init__(self) -> None:
        for name in (
            "width",
            "heads",
            "graph_layers",
            "latent_rank",
            "streams",
            "projector_hidden",
            "projector_bottleneck",
            "prototypes",
            "sparse_topk",
            "sparse_index_dim",
            "sparse_query_chunk",
            "kda_layers",
            "relay_scan_chunk_size",
            "relay_graph_chunk_rows",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "prior_shared_adapter",
            "gene_conditioned_readout",
            "direct_target_flag",
            "learned_genept_projection",
            "checkpoint_genept_mapping",
            "graph_mhc",
            "all_kda_prefix",
            "fixed_gene_order",
            "response_injection_residual",
            "ssl2_cls_input_layernorm",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (
            any(
                (
                    self.prior_shared_adapter,
                    self.gene_conditioned_readout,
                    self.direct_target_flag,
                    self.learned_genept_projection,
                )
            )
            and self.attention != "relay_full"
        ):
            raise ValueError("unseen-gene mechanisms require the relay profile")
        if self.learned_genept_projection and (
            self.prior_shared_adapter
            or (self.gene_conditioned_readout and self.mlp_profile not in ("unified", "restored"))
        ):
            raise ValueError("raw GenePT projection cannot use reduced-prior mechanisms")
        if self.checkpoint_genept_mapping and not self.learned_genept_projection:
            raise ValueError("GenePT mapping checkpoint requires the learned raw-prior projection")
        if self.mlp_profile not in ("historical", "unified", "restored"):
            raise ValueError("unknown MLP profile")
        if (
            type(self.ffn_clip) is not bool
            or self.ffn_variant not in ("mixed", "swiglu", "gelu")
            or self.ffn_input_norm not in ("historical", "rmsnorm", "layernorm")
        ):
            raise ValueError("invalid restored FFN controls")
        if (
            self.ffn_linear_bias not in ("historical", "none")
            or type(self.prototype_weight_norm) is not bool
            or type(self.prototype_fixed_scale) is not bool
            or type(self.prototype_freeze_gradients) is not bool
            or type(self.prototype_freeze_epochs) is not int
            or self.prototype_freeze_epochs < 0
            or self.prediction_parameterization not in ("residual", "softplus")
        ):
            raise ValueError("invalid five-arm FFN controls")
        if self.ssl2_cls_input_layernorm and (
            self.mlp_profile != "restored" or self.ffn_input_norm != "historical"
        ):
            raise ValueError(
                "SSL2 CLS input LayerNorm requires restored profile and historical FFN input norm"
            )
        if self.prototype_freeze_epochs and not self.prototype_weight_norm:
            raise ValueError("prototype freeze requires the explicit weight-normalized arm")
        if self.prototype_fixed_scale and not self.prototype_weight_norm:
            raise ValueError("fixed prototype scale requires weight normalization")
        if self.prototype_freeze_gradients and not self.prototype_freeze_epochs:
            raise ValueError("gradient freeze requires a positive prototype freeze budget")
        if (
            self.ffn_linear_bias != "historical"
            or self.prototype_weight_norm
            or self.prototype_freeze_epochs
            or self.prediction_parameterization != "residual"
        ) and self.mlp_profile != "restored":
            raise ValueError("five-arm FFN controls require restored profile")
        if self.mlp_profile == "restored":
            if (
                self.attention != "relay_full"
                or not self.learned_genept_projection
                or self.genept_projection_activation != "gelu"
                or any(
                    type(v) is not int or v < 1
                    for v in (self.genept_hidden_width, self.genept_middle_width)
                )
            ):
                raise ValueError(
                    "restored FFN requires relay/raw shared GELU GenePT and positive widths"
                )
        elif (
            self.ffn_variant != "mixed"
            or self.ffn_input_norm != "historical"
            or not self.ffn_clip
            or self.genept_hidden_width != 0
            or self.genept_middle_width != 0
        ):
            raise ValueError("new FFN controls require restored profile")
        if self.perturbation_injection not in ("per_layer", "entry"):
            raise ValueError("unknown perturbation injection")
        if any(
            type(v) is not bool for v in (self.control_conditioned_graph, self.random_gene_order)
        ):
            raise ValueError("graph conditioning and gene order must be boolean")
        if (
            self.mlp_profile == "unified"
            or self.control_conditioned_graph
            or self.perturbation_injection != "per_layer"
            or self.response_injection_residual
            or not self.random_gene_order
        ) and self.attention != "relay_full":
            raise ValueError("new mechanism profiles require relay attention")
        if self.control_conditioned_graph and (
            self.mlp_profile not in ("unified", "restored") or not self.graph_source_key_gate
        ):
            raise ValueError(
                "conditional graph requires an explicit MLP profile and source key gate"
            )
        if self.mlp_profile == "unified" and self.genept_projection_activation != "none":
            raise ValueError("unified MLP replaces the historical GenePT activation")
        if self.genept_projection_activation not in ("none", "gelu"):
            raise ValueError("unknown GenePT projection activation")
        if self.genept_projection_activation != "none" and not self.learned_genept_projection:
            raise ValueError("GenePT projection activation requires learned GenePT projection")
        if self.sinkhorn_backend not in ("native", "auto", "triton"):
            raise ValueError("unknown Sinkhorn backend")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("invalid head width or dropout")
        if self.attention not in (
            "hybrid",
            "hybrid_sparse",
            "full_latent",
            "delta_full",
            "full",
            "per_gene",
            "relay_full",
        ):
            raise ValueError("unknown attention variant")
        if self.ffn_type not in ("gelu", "swiglu"):
            raise ValueError("unknown FFN type")
        if self.graph_read_mode not in ("static", "propagated", "relay"):
            raise ValueError("unknown graph read mode")
        if self.graph_read_mode == "propagated" and self.graph_layers != 2:
            raise ValueError("propagated graph read currently requires two layers")
        if self.graph_read_mode == "relay" and self.graph_layers != 4:
            raise ValueError("relay graph read requires three KDA plus one MLA layer")
        if self.graph_mhc and (self.graph_read_mode != "relay" or self.control_conditioned_graph):
            raise ValueError("graph mHC requires relay graph without control conditioning")
        if self.attention == "relay_full" and (
            self.kda_layers != 2 or self.graph_read_mode != "relay"
        ):
            raise ValueError("relay profile requires two cell/response KDA and relay graph")
        if self.sparse_topk < 2:
            raise ValueError("sparse_topk must leave slots for self and CLS")
        if type(self.checkpoint_layers) is not bool:
            raise ValueError("checkpoint_layers must be boolean")
        if self.relay_eval_seed is not None and (
            type(self.relay_eval_seed) is not int or self.relay_eval_seed < 0
        ):
            raise ValueError("relay evaluation seed must be a nonnegative integer")
        if type(self.relay_passes) is not int or self.relay_passes not in (1, 2):
            raise ValueError("relay_passes must be 1 or 2")
        if self.relay_passes != 2 and self.attention != "relay_full":
            raise ValueError("single-pass final-state KDA requires the relay profile")
        if self.relay_kernel not in ("eager", "inductor", "cudagraphs"):
            raise ValueError("unknown relay kernel")
        if self.weighted_gram_backend not in ("eager", "triton_io"):
            raise ValueError("unknown weighted Gram backend")
        if self.weighted_gram_backend != "eager" and (
            self.attention != "relay_full"
            or self.relay_kernel != "eager"
            or self.attention_replacement != "none"
            or self.self_readout != "final"
            or self.all_kda_prefix
            or self.short_graph_kernel
            or self.width // self.heads != 64
            or self.relay_scan_chunk_size > 32
            or (self.relay_sequence_chunk_size or self.relay_scan_chunk_size) > 32
        ):
            raise ValueError("Gram IO requires final-state eager relay W64 and chunks at most32")
        if self.relay_kernel != "eager" and self.attention != "relay_full":
            raise ValueError("compiled relay kernel requires the relay profile")
        if type(self.relay_validate_once) is not bool:
            raise ValueError("relay_validate_once must be boolean")
        if self.relay_validate_once and self.graph_read_mode != "relay":
            raise ValueError("single neighborhood validation requires relay graph")
        if type(self.short_graph_kernel) is not bool:
            raise ValueError("short graph kernel must be boolean")
        if self.short_graph_kernel and (
            self.attention != "relay_full"
            or self.graph_read_mode != "relay"
            or self.relay_passes != 1
            or self.relay_kernel != "eager"
            or self.width // self.heads != 64
        ):
            raise ValueError("short graph kernel requires W64 single-pass eager relay graph")
        if type(self.cache_kda_constants) is not bool:
            raise ValueError("KDA constant cache must be boolean")
        if type(self.graph_source_key_gate) is not bool:
            raise ValueError("graph source key gate must be boolean")
        if self.graph_source_key_gate and self.graph_read_mode != "relay":
            raise ValueError("graph source key gate requires four-layer relay graph")
        if self.attention_replacement not in ("none", "softmax", "retention"):
            raise ValueError("unknown complete attention replacement")
        if self.self_readout not in ("final", "position"):
            raise ValueError("unknown self-KDA readout")
        if (self.attention_replacement != "none" or self.self_readout != "final") and (
            self.attention != "relay_full" or self.relay_kernel != "eager"
        ):
            raise ValueError("functional attention ablations require eager relay architecture")
        if self.self_readout == "position" and (
            self.relay_passes != 1 or self.attention_replacement != "none"
        ):
            raise ValueError("position readout requires single-pass self-KDA")
        if self.fixed_gene_order and (self.random_gene_order or self.attention != "relay_full"):
            raise ValueError("fixed ID order requires nonrandom relay attention")
        if self.all_kda_prefix and (
            not self.fixed_gene_order
            or self.self_readout != "final"
            or self.relay_passes != 1
            or self.relay_kernel != "eager"
            or self.attention_replacement != "none"
            or self.short_graph_kernel
            or self.graph_mhc
        ):
            raise ValueError("all-KDA prefix requires fixed ID order, single-pass eager KDA")
        if self.attention_replacement != "none" and (
            self.short_graph_kernel or self.cache_kda_constants or self.relay_passes != 1
        ):
            raise ValueError("replacement cannot enable KDA-only execution changes")
        if self.relay_sequence_chunk_size is not None and (
            type(self.relay_sequence_chunk_size) is not int or self.relay_sequence_chunk_size <= 0
        ):
            raise ValueError("relay sequence chunk size must be a positive integer")
        if self.cache_kda_constants and (
            self.attention != "relay_full" or self.relay_kernel != "eager"
        ):
            raise ValueError("KDA constant cache requires eager relay attention")
        if (
            self.relay_scan_chunk_size != 32
            or self.relay_graph_chunk_rows != 64
            or self.relay_sequence_chunk_size is not None
        ) and (self.attention != "relay_full" or self.relay_kernel != "eager"):
            raise ValueError("relay chunk tuning requires eager relay attention")

    @classmethod
    def parse(cls, values: dict[str, Any]) -> V2Architecture:
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown v2 architecture fields: {sorted(unknown)}")
        return cls(**values)

    def payload(self) -> dict[str, Any]:
        values = asdict(self)
        if self.sinkhorn_backend == "native":
            values.pop("sinkhorn_backend")
        if self.relay_eval_seed is None:
            # Preserve architecture identity of checkpoints predating relay KDA.
            values.pop("relay_eval_seed")
        if self.relay_passes == 2:
            # Missing field retains historical two-pass checkpoint identity.
            values.pop("relay_passes")
        if self.relay_kernel == "eager":
            values.pop("relay_kernel")
        if self.weighted_gram_backend == "eager":
            values.pop("weighted_gram_backend")
        if not self.relay_validate_once:
            values.pop("relay_validate_once")
        if not self.short_graph_kernel:
            values.pop("short_graph_kernel")
        if not self.cache_kda_constants:
            values.pop("cache_kda_constants")
        if self.relay_scan_chunk_size == 32:
            values.pop("relay_scan_chunk_size")
        if self.relay_graph_chunk_rows == 64:
            values.pop("relay_graph_chunk_rows")
        if self.relay_sequence_chunk_size is None:
            values.pop("relay_sequence_chunk_size")
        if not self.graph_source_key_gate:
            values.pop("graph_source_key_gate")
        if not self.graph_mhc:
            values.pop("graph_mhc")
        if not self.checkpoint_genept_mapping:
            values.pop("checkpoint_genept_mapping")
        if self.attention_replacement == "none":
            values.pop("attention_replacement")
        if self.self_readout == "final":
            values.pop("self_readout")
        for name in ("all_kda_prefix", "fixed_gene_order"):
            if not getattr(self, name):
                values.pop(name)
        if self.genept_projection_activation == "none":
            # Preserve the architecture identity of historical linear U4 checkpoints.
            values.pop("genept_projection_activation")
        for name, default in (
            ("mlp_profile", "historical"),
            ("ffn_variant", "mixed"),
            ("ffn_input_norm", "historical"),
            ("ssl2_cls_input_layernorm", False),
            ("ffn_clip", True),
            ("ffn_linear_bias", "historical"),
            ("prototype_weight_norm", False),
            ("prototype_freeze_epochs", 0),
            ("prototype_fixed_scale", False),
            ("prototype_freeze_gradients", False),
            ("prediction_parameterization", "residual"),
            ("genept_hidden_width", 0),
            ("genept_middle_width", 0),
            ("control_conditioned_graph", False),
            ("perturbation_injection", "per_layer"),
            ("response_injection_residual", False),
            ("random_gene_order", True),
        ):
            if getattr(self, name) == default:
                values.pop(name)
        for name in (
            "prior_shared_adapter",
            "gene_conditioned_readout",
            "direct_target_flag",
            "learned_genept_projection",
        ):
            if not getattr(self, name):
                values.pop(name)
        return values


@dataclass(frozen=True)
class V2Options:
    query_count: int
    microbatch: int
    accumulation: int
    world_size: int
    eval_query_count: int
    graph_expander_degree: int
    graph_expander_seed: int
    lambda1: float
    lambda2: float
    ssl1_condition: float
    ssl1_node: float
    ssl1_spread: float
    ssl2_dino: float
    ssl2_ibot: float
    ssl2_koleo: float
    ssl1_reduction: str
    local_views: int
    global_min_ratio: float
    global_max_ratio: float
    local_min_ratio: float
    local_max_ratio: float
    mask_probability: float
    mask_min_ratio: float
    mask_max_ratio: float
    graph_mask_ratio: float
    graph_edge_dropout: float
    max_conditions: int
    teacher_start: float
    teacher_end: float
    genept_artifact_path: str
    genept_sha256: str
    graph_manifest_path: str
    graph_manifest_sha256: str
    gene_initialization: str
    prediction_loss: str
    loss_reduction: str = "row_mean"
    graph_expander_type: str = "permutation"
    expression_holdout_path: str = ""
    expression_holdout_sha256: str = ""
    koleo_exclude_same_condition: bool = False
    graph_view_mode: str = "legacy"
    validation_mode: str = "joint_and_prediction"
    train_selection_path: str = ""
    train_selection_sha256: str = ""
    prediction_error_power: int = 2
    prediction_reduction_override: str = "inherit"
    auxiliary_mask_ratio: float = 0.0
    lambda_gene_mask: float = 0.0
    lambda_cls_mask: float = 0.0
    ssl1_local_views: int | None = None
    ssl2_local_views: int | None = None
    ssl1_local_layout: str = "go_string"
    independent_view_rng: bool = False
    masked_response_ratio: float = 0.0
    lambda_masked_response: float = 0.0
    population_response: bool = False
    lambda_mmd: float = 1.0
    process_diagnostics: bool = False
    teacher_temperature_start: float = 0.04
    teacher_temperature_end: float = 0.04
    teacher_temperature_warmup_epochs: int = 0
    training_mode: str = "joint"
    readout_parent_run_root: str = ""
    readout_parent_checkpoint_sha256: str = ""
    readout_parent_source_commit: str = ""
    readout_parent_run_id: str = ""
    readout_parent_epoch: int = 0
    ssl_pretrain_epochs: int = 0
    finetune_learning_rate: float = 2e-4
    au_alignment_weight: float = 0.0
    au_uniformity_weight: float = 0.0
    au_temperature: float = 2.0
    au_warmup_steps: int = 30
    cr_weight: float = 0.0
    cr_epsilon_squared: float = 0.05
    cr_warmup_steps: int = 30
    population_aux_space: str = "response_cls"
    spectral_rank_target: float = 0.20
    spectral_rank_weight: float = 0.0
    spectral_cosine_weight: float = 0.0
    spectral_softness: float = 0.25
    spectral_warmup_steps: int = 30
    spectral_amplitude_target: float = 0.0225
    spectral_amplitude_weight: float = 0.0

    @classmethod
    def parse_parameters(cls, values: dict[str, Any]) -> tuple[V2Architecture, V2Options]:
        arch_names = {f.name for f in fields(V2Architecture)}
        names = {f.name for f in fields(cls)}
        optional = {
            "sinkhorn_backend",
            "expression_holdout_path",
            "expression_holdout_sha256",
            "loss_reduction",
            "ffn_type",
            "sparse_topk",
            "sparse_index_dim",
            "sparse_query_chunk",
            "kda_layers",
            "graph_read_mode",
            "graph_expander_type",
            "relay_eval_seed",
            "relay_passes",
            "relay_kernel",
            "weighted_gram_backend",
            "relay_validate_once",
            "short_graph_kernel",
            "cache_kda_constants",
            "relay_scan_chunk_size",
            "relay_graph_chunk_rows",
            "relay_sequence_chunk_size",
            "graph_source_key_gate",
            "graph_mhc",
            "attention_replacement",
            "self_readout",
            "all_kda_prefix",
            "fixed_gene_order",
            "prior_shared_adapter",
            "gene_conditioned_readout",
            "direct_target_flag",
            "learned_genept_projection",
            "checkpoint_genept_mapping",
            "genept_projection_activation",
            "koleo_exclude_same_condition",
            "graph_view_mode",
            "validation_mode",
            "train_selection_path",
            "train_selection_sha256",
            "prediction_error_power",
            "prediction_reduction_override",
            "auxiliary_mask_ratio",
            "lambda_gene_mask",
            "lambda_cls_mask",
            "mlp_profile",
            "ffn_variant",
            "ffn_input_norm",
            "ffn_clip",
            "ffn_linear_bias",
            "prototype_weight_norm",
            "prototype_freeze_epochs",
            "prototype_fixed_scale",
            "prototype_freeze_gradients",
            "prediction_parameterization",
            "genept_hidden_width",
            "genept_middle_width",
            "control_conditioned_graph",
            "perturbation_injection",
            "response_injection_residual",
            "ssl2_cls_input_layernorm",
            "random_gene_order",
            "ssl1_local_views",
            "ssl2_local_views",
            "ssl1_local_layout",
            "independent_view_rng",
            "masked_response_ratio",
            "lambda_masked_response",
            "population_response",
            "lambda_mmd",
            "process_diagnostics",
            "teacher_temperature_start",
            "teacher_temperature_end",
            "teacher_temperature_warmup_epochs",
            "training_mode",
            "readout_parent_run_root",
            "readout_parent_checkpoint_sha256",
            "readout_parent_source_commit",
            "readout_parent_run_id",
            "readout_parent_epoch",
            "ssl_pretrain_epochs",
            "finetune_learning_rate",
            "au_alignment_weight",
            "au_uniformity_weight",
            "au_temperature",
            "au_warmup_steps",
            "cr_weight",
            "cr_epsilon_squared",
            "cr_warmup_steps",
            "population_aux_space",
            "spectral_rank_target",
            "spectral_rank_weight",
            "spectral_cosine_weight",
            "spectral_softness",
            "spectral_warmup_steps",
            "spectral_amplitude_target",
            "spectral_amplitude_weight",
        }
        required = (arch_names | names) - optional
        if not required <= set(values) or set(values) - (arch_names | names):
            raise ValueError(
                f"v2 missing fields={sorted(required - set(values))}; "
                f"unknown={sorted(set(values) - (arch_names | names))}"
            )
        plain = {name: value.value for name, value in values.items()}
        arch = V2Architecture.parse({name: plain[name] for name in arch_names if name in plain})
        options = cls(**{name: plain[name] for name in names if name in plain})
        if (
            arch.prior_shared_adapter
            or arch.gene_conditioned_readout
            or arch.learned_genept_projection
        ) and (options.gene_initialization != "genept"):
            raise ValueError("prior mechanisms require GenePT initialization")
        return arch, options

    def __post_init__(self) -> None:
        for name in (
            "spectral_rank_weight",
            "spectral_cosine_weight",
            "spectral_rank_target",
            "spectral_amplitude_target",
            "spectral_amplitude_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid spectral parameter: {name}")
        if (
            not 0 < self.spectral_rank_target <= 1
            or not 0 < self.spectral_amplitude_target <= 1
            or isinstance(self.spectral_softness, bool)
            or self.spectral_softness != 0.25
            or type(self.spectral_warmup_steps) is not int
            or self.spectral_warmup_steps < 0
        ):
            raise ValueError("spectral target needs (0,1], softness .25, nonnegative warmup")
        if (
            self.spectral_rank_weight
            or self.spectral_cosine_weight
            or self.spectral_amplitude_weight
        ) and (
            not self.lambda2
            or self.loss_reduction != "row_mean"
            or self.ssl2_koleo
            or self.au_alignment_weight
            or self.au_uniformity_weight
            or self.cr_weight
            or self.population_response
            or self.population_aux_space != "response_cls"
            or self.training_mode != "joint"
            or self.max_conditions == 1
        ):
            raise ValueError(
                "spectral pair requires joint raw Response CLS, row_mean "
                "and disabled AU/CR/KoLeo/population loss"
            )
        if self.population_aux_space not in ("response_cls", "ssl2_cls_bottleneck"):
            raise ValueError("unknown population auxiliary space")
        if (
            isinstance(self.cr_weight, bool)
            or not isinstance(self.cr_weight, (int, float))
            or not math.isfinite(self.cr_weight)
            or self.cr_weight < 0
            or isinstance(self.cr_epsilon_squared, bool)
            or self.cr_epsilon_squared != 0.05
            or type(self.cr_warmup_steps) is not int
            or self.cr_warmup_steps < 0
        ):
            raise ValueError(
                "CR requires finite nonnegative weight, epsilon_squared .05 and nonnegative warmup"
            )
        if self.cr_weight and (
            not self.lambda2
            or self.loss_reduction != "row_mean"
            or self.ssl2_koleo
            or self.population_response
            or self.training_mode not in ("joint", "ssl_then_joint")
            or self.max_conditions == 1
        ):
            raise ValueError(
                "CR requires mixed joint SSL2 row_mean and disabled KoLeo/population loss"
            )
        for name in ("au_alignment_weight", "au_uniformity_weight", "au_temperature"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid AU parameter: {name}")
        if (
            self.au_temperature <= 0
            or type(self.au_warmup_steps) is not int
            or self.au_warmup_steps < 0
        ):
            raise ValueError("AU temperature must be positive and warmup nonnegative")
        if (self.au_alignment_weight or self.au_uniformity_weight) and (
            not self.lambda2
            or self.loss_reduction != "row_mean"
            or self.ssl2_koleo
            or self.population_response
            or self.training_mode not in ("joint", "ssl_then_joint")
            or self.max_conditions == 1
        ):
            raise ValueError(
                "AU requires mixed joint SSL2, row_mean, and disabled KoLeo/population loss"
            )
        if self.training_mode not in ("joint", "ssl_then_mse", "ssl_then_joint", "frozen_readout"):
            raise ValueError("unknown v2 training mode")
        parent_fields = (
            self.readout_parent_run_root,
            self.readout_parent_checkpoint_sha256,
            self.readout_parent_source_commit,
            self.readout_parent_run_id,
        )
        if self.training_mode == "frozen_readout":
            if (
                not all(parent_fields)
                or self.readout_parent_epoch != 3
                or len(self.readout_parent_checkpoint_sha256) != 64
                or len(self.readout_parent_source_commit) != 40
                or self.lambda1
                or self.lambda2
                or self.au_alignment_weight
                or self.au_uniformity_weight
                or self.cr_weight
                or self.population_response
                or self.auxiliary_mask_ratio
                or self.masked_response_ratio
                or self.validation_mode != "disabled"
                or self.prediction_error_power != 2
                or self.loss_reduction != "row_mean"
            ):
                raise ValueError("frozen readout requires a sealed R9 parent and MSE-only row mean")
        elif any(parent_fields) or self.readout_parent_epoch:
            raise ValueError("readout parent fields require frozen_readout mode")
        if type(self.ssl_pretrain_epochs) is not int or (
            self.ssl_pretrain_epochs < 0
            or (self.training_mode in ("ssl_then_mse", "ssl_then_joint"))
            != bool(self.ssl_pretrain_epochs)
        ):
            raise ValueError("two-stage training needs an explicit pretraining epoch budget")
        if (
            isinstance(self.finetune_learning_rate, bool)
            or not math.isfinite(self.finetune_learning_rate)
            or self.finetune_learning_rate <= 0
        ):
            raise ValueError("finetuning learning rate must be positive and finite")
        for value in (self.teacher_temperature_start, self.teacher_temperature_end):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not (math.isfinite(value) and value > 0)
            ):
                raise ValueError("teacher temperatures must be finite and positive")
        if (
            type(self.teacher_temperature_warmup_epochs) is not int
            or self.teacher_temperature_warmup_epochs < 0
            or bool(self.teacher_temperature_warmup_epochs)
            != (self.teacher_temperature_start != self.teacher_temperature_end)
        ):
            raise ValueError("temperature transition needs a positive epoch budget")
        if type(self.process_diagnostics) is not bool:
            raise ValueError("process diagnostics must be boolean")
        if self.ssl1_local_layout not in ("go_string", "go_string_random", "go_go_string_string"):
            raise ValueError("unknown SSL1 Local layout")
        for value in (self.ssl1_local_views, self.ssl2_local_views):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("branch Local count must be positive")
        if (
            self.ssl1_local_layout != "go_string"
            and (self.ssl1_local_views or self.local_views) != 4
        ):
            raise ValueError("four-Local layouts require four SSL1 Locals")
        if (
            type(self.independent_view_rng) is not bool
            or type(self.population_response) is not bool
        ):
            raise ValueError("view RNG and population response must be boolean")
        if not all(
            math.isfinite(v)
            for v in (self.masked_response_ratio, self.lambda_masked_response, self.lambda_mmd)
        ):
            raise ValueError("new supervision settings must be finite")
        if (
            not 0 <= self.masked_response_ratio < 1
            or min(self.lambda_masked_response, self.lambda_mmd) < 0
        ):
            raise ValueError("invalid masked-response or MMD setting")
        if bool(self.masked_response_ratio) != bool(self.lambda_masked_response):
            raise ValueError("masked response needs a mask ratio and a positive weight")
        if type(self.prediction_error_power) is not int or self.prediction_error_power not in (
            2,
            4,
        ):
            raise ValueError("prediction error power must be 2 or 4")
        if self.prediction_reduction_override not in ("inherit", "row_mean", "condition_mean"):
            raise ValueError("unknown prediction-only reduction")
        if not all(
            math.isfinite(x)
            for x in (self.auxiliary_mask_ratio, self.lambda_gene_mask, self.lambda_cls_mask)
        ):
            raise ValueError("reconstruction settings must be finite")
        if (
            not 0 <= self.auxiliary_mask_ratio < 1
            or min(self.lambda_gene_mask, self.lambda_cls_mask) < 0
        ):
            raise ValueError("invalid control reconstruction mask or weights")
        if bool(self.auxiliary_mask_ratio) != bool(self.lambda_gene_mask or self.lambda_cls_mask):
            raise ValueError("control reconstruction mask and supervision must be enabled together")
        if self.loss_reduction not in ("row_mean", "condition_mean"):
            raise ValueError("unknown unified loss reduction")
        if self.graph_expander_type not in ("permutation", "hamiltonian"):
            raise ValueError("unknown graph expander type")
        if type(self.koleo_exclude_same_condition) is not bool:
            raise ValueError("KoLeo condition exclusion must be boolean")
        if self.graph_view_mode not in ("legacy", "multiscale"):
            raise ValueError("unknown graph view mode")
        if self.validation_mode not in ("joint_and_prediction", "joint_only", "disabled"):
            raise ValueError("unknown v2 validation mode")
        if bool(self.train_selection_path) != bool(self.train_selection_sha256):
            raise ValueError("training row selection path and hash must be supplied together")
        if self.train_selection_sha256 and (
            len(self.train_selection_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.train_selection_sha256)
        ):
            raise ValueError("training row selection requires a lowercase SHA256")
        if bool(self.expression_holdout_path) != bool(self.expression_holdout_sha256):
            raise ValueError("expression holdout manifest path and hash must be supplied together")
        if self.expression_holdout_sha256 and (
            len(self.expression_holdout_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.expression_holdout_sha256)
        ):
            raise ValueError("expression holdout requires a lowercase SHA256")

        for name in (
            "query_count",
            "microbatch",
            "accumulation",
            "world_size",
            "eval_query_count",
            "local_views",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_conditions) is not int or self.max_conditions < 0:
            raise ValueError("max_conditions must be nonnegative (zero means uncapped)")
        for name in (
            "lambda1",
            "lambda2",
            "ssl1_condition",
            "ssl1_node",
            "ssl1_spread",
            "ssl2_dino",
            "ssl2_ibot",
            "ssl2_koleo",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= value < float("inf")
            ):
                raise ValueError(f"invalid loss weight: {name}")
        for name in ("global", "local", "mask"):
            lo, hi = getattr(self, name + "_min_ratio"), getattr(self, name + "_max_ratio")
            if not 0 < lo <= hi <= 1:
                raise ValueError(f"invalid {name} ratio interval")
        if not 0 <= self.teacher_start <= self.teacher_end <= 1:
            raise ValueError("invalid EMA schedule")
        if self.ssl1_reduction not in ("condition_mean", "row_mean"):
            raise ValueError("invalid SSL1 reduction")
        if self.prediction_loss not in ("cell_mean", "condition_mean"):
            raise ValueError("invalid prediction loss")
        if self.gene_initialization not in ("genept", "random"):
            raise ValueError("invalid gene initialization")
        if self.max_conditions == 1 and self.ssl2_koleo != 0:
            raise ValueError("single-condition batches require SSL2 KoLeo disabled")
        for name in ("mask_probability", "graph_mask_ratio", "graph_edge_dropout"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"invalid probability: {name}")
        for name in ("genept_sha256", "graph_manifest_sha256"):
            digest = getattr(self, name)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name} must be an exact SHA256")
