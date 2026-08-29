from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/performance/preflight_ablation_census.py"
MATRIX = PROJECT_ROOT / "configs/ablations/nadig_jurkat/matrix.json"
SOURCE_COMMIT = "a" * 40
HASH = "b" * 64


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("preflight_ablation_census_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load P0 census preflight")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preflight() -> ModuleType:
    return _load_script()


def _clean_runtime_import_snapshot() -> dict[str, object]:
    return {
        "measurement_method": "sys.modules_snapshot_without_importing_torch",
        "torch_loaded": False,
        "torch_cuda_loaded": False,
        "torch_module_count": 0,
        "torch_cuda_module_count": 0,
        "torch_modules": [],
        "torch_cuda_modules": [],
    }


@pytest.fixture(autouse=True)
def _isolate_functional_runtime_import_guard(
    preflight: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep in-process P0 fixtures independent of unrelated test imports."""

    monkeypatch.setattr(
        preflight,
        "_runtime_import_snapshot",
        lambda: _clean_runtime_import_snapshot(),
    )
    monkeypatch.setattr(
        preflight,
        "_require_no_torch_imports",
        lambda _stage: _clean_runtime_import_snapshot(),
    )


@pytest.fixture(scope="module")
def matrix_sha256() -> str:
    return hashlib.sha256(MATRIX.read_bytes()).hexdigest()


def _data_identity(preflight: ModuleType):
    return preflight.DataIdentity(
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        canonical_manifest_path="/data/canonical.json",
        canonical_manifest_sha256="1" * 64,
        canonical_manifest_size_bytes=101,
        canonical_data_path="/data/canonical.h5ad",
        canonical_data_sha256="2" * 64,
        canonical_data_size_bytes=202,
        observation_order_sha256="7" * 64,
        split_manifest_path="/data/split.json",
        split_manifest_sha256="3" * 64,
        split_manifest_size_bytes=303,
        split_content_sha256="4" * 64,
        source_manifest_path="/data/source.json",
        source_manifest_sha256="5" * 64,
        source_manifest_size_bytes=404,
        source_h5ad_path="/data/source.h5ad",
        source_h5ad_sha256="6" * 64,
        source_h5ad_size_bytes=505,
    )


def _manifest(hvg_count: int):
    direct = [f"G{index}" for index in range(hvg_count)]
    targets = [f"T{index}" for index in range(10)]
    graph = [*direct, *targets]
    graph_hash = hashlib.sha256(
        json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target_hash = hashlib.sha256(
        json.dumps(targets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SimpleNamespace(
        dataset_id="nadig_jurkat",
        protocol_id="within_cell_unseen_single",
        canonical_data_sha256="2" * 64,
        split_content_sha256="4" * 64,
        source_h5ad_sha256="6" * 64,
        source_registry_sha256="7" * 64,
        hvg_method="scanpy.pp.highly_variable_genes",
        hvg_flavor="seurat",
        normalize_total=4000,
        log1p=True,
        hvg_subset=True,
        requested_hvg_count=hvg_count,
        expression_gene_count=5000,
        hvg_fit_scope="full_filtered_cell_line_pre_split",
        hvg_fit_cell_count=1000,
        hvg_fit_condition_ids=["CTRL", "PERT"],
        hvg_fit_condition_ids_sha256="8" * 64,
        direct_hvg_gene_ids=direct,
        normalized_dispersion_ranked_hvg_gene_ids=direct,
        candidate_target_ids=targets,
        candidate_target_order_sha256=target_hash,
        graph_gene_ids=graph,
        graph_gene_order_sha256=graph_hash,
        graph_gene_count=len(graph),
        source_artifact_sha256={"go": "9" * 64, "string": "a" * 64},
        source_pruned_nonself_edge_count={"go": 100, "string": 120},
        topology_content_sha256=(f"{hvg_count:064x}"[-64:]),
        gene_feature_policy="learned_id",
    )


def _config_runtime_roots(preflight: ModuleType, matrix_sha256: str) -> dict[int, str]:
    roots: dict[int, str] = {}
    for binding in preflight.census.bind_matrix_variants(
        MATRIX,
        repository_root=PROJECT_ROOT,
        expected_matrix_sha256=matrix_sha256,
    ):
        config = preflight.load_experiment_config(binding.config_path)
        hvg = int(config.model.parameters["graph_hvg_count"].value)
        roots[hvg] = str(config.model.parameters["runtime_graph_root"].value)
    return roots


def _write_graph_slots(
    preflight: ModuleType,
    data_root: Path,
    matrix_sha256: str,
    *,
    missing_hvg: int | None = None,
    tampered_source_hvg: int | None = None,
    escaping_source_hvg: int | None = None,
) -> tuple[dict[str, tuple[object, object]], dict[int, str]]:
    roots = _config_runtime_roots(preflight, matrix_sha256)
    loaded: dict[str, tuple[object, object]] = {}
    for hvg_count, relative in roots.items():
        if hvg_count == missing_hvg:
            continue
        root = data_root / relative
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps({"hvg_count": hvg_count}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = _manifest(hvg_count)
        (root / "graph_gene_ids.txt").write_text(
            "\n".join(manifest.graph_gene_ids) + "\n",
            encoding="utf-8",
        )
        (root / f"hvg{hvg_count}_dispersion_ranking.json").write_text(
            json.dumps({"hvg_count": hvg_count, "ordered_entries": []}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for source_name in ("go", "string"):
            artifact_path = root / f"{source_name}.npz"
            artifact_path.write_bytes(f"{hvg_count}:{source_name}:sealed\n".encode())
            manifest.source_artifact_sha256[source_name] = hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest()
        if hvg_count == tampered_source_hvg:
            (root / "go.npz").write_bytes(b"tampered after manifest identity\n")
        if hvg_count == escaping_source_hvg:
            outside = data_root.parent / f"outside-{hvg_count}-go.npz"
            outside.write_bytes((root / "go.npz").read_bytes())
            (root / "go.npz").unlink()
            (root / "go.npz").symlink_to(outside)
        loaded[str(root.resolve())] = (
            SimpleNamespace(gene_ids=tuple(manifest.graph_gene_ids)),
            manifest,
        )
    return loaded, roots


def _genept_receipt(
    preflight: ModuleType,
    *,
    data_root: Path,
    graph_relative: str,
    manifest,
    wrong_topology: bool = False,
    selected_count_delta: int = 0,
) -> Path:
    config = preflight.load_experiment_config(
        PROJECT_ROOT
        / "configs/ablations/nadig_jurkat/e1_frozen_genept/gradpert_b2/nadig_jurkat.yaml"
    )
    graph_manifest_path = data_root / graph_relative / "manifest.json"
    topology = "f" * 64 if wrong_topology else manifest.topology_content_sha256
    selected_count = manifest.graph_gene_count + selected_count_delta
    source_count = 17730
    receipt = preflight.GenePTSeedAvailabilityReceipt(
        schema_version="genept-seed-go-protein-pathway-availability-v1",
        status="available",
        dataset_id="nadig_jurkat",
        identifier_matching="exact_case_sensitive",
        extra_source_gene_policy="ignore_preserving_runtime_axis",
        missing_runtime_gene_policy="fail_before_model_construction",
        missing_perturbation_target_policy="fail_before_model_construction",
        parent_topology_content_sha256=topology,
        parent_graph_gene_order_sha256=manifest.graph_gene_order_sha256,
        candidate_target_order_sha256=manifest.candidate_target_order_sha256,
        prior_contract_id="seed_go_protein_pathway_master_v1",
        runtime_graph_root=graph_relative,
        parent_graph_manifest_sha256=hashlib.sha256(graph_manifest_path.read_bytes()).hexdigest(),
        genept_source_path=config.model.parameters["genept_artifact_path"].value,
        genept_source_size_bytes=123,
        genept_source_sha256=config.model.parameters["genept_expected_sha256"].value,
        genept_model="doubao-embedding-vision",
        embedding_width=2048,
        source_gene_count=source_count,
        source_gene_order_sha256="c" * 64,
        selected_gene_count=selected_count,
        selected_gene_order_sha256=manifest.graph_gene_order_sha256,
        selected_matrix_sha256="d" * 64,
        extra_source_gene_count=source_count - selected_count,
        extra_source_gene_ids_sha256="e" * 64,
        perturbation_target_gene_count=len(manifest.candidate_target_ids),
        perturbation_target_gene_ids_sha256=manifest.candidate_target_order_sha256,
        zero_vector_gene_count=0,
        result_topology_content_sha256=topology,
    )
    path = data_root / (
        "genept-wrong.json" if wrong_topology or selected_count_delta else "genept.json"
    )
    path.write_text(receipt.model_dump_json(), encoding="utf-8")
    return path


def _dependencies(
    preflight: ModuleType,
    loaded: dict[str, tuple[object, object]],
    *,
    bad_hvg: int | None = None,
):
    def load_graph(path: Path):
        if bad_hvg is not None and f"hvg{bad_hvg}_plus_targets" in str(path):
            raise ValueError("vNext graph artifact hash mismatch: string")
        return loaded[str(path.resolve())]

    return preflight.PreflightDependencies(
        inspect_source=lambda root, commit, publication, publication_sha: {
            "schema_version": "nadig-vnext-performance-source-identity-v1",
            "repository_root": str(root.resolve()),
            "expected_repository": preflight.EXPECTED_REPOSITORY,
            "expected_commit": commit,
            "observed_commit": commit,
            "git_tree_object": "c" * 40,
            "source_tree_sha256": "d" * 64,
            "source_tree_identity_method": ("gradpert.execution.identity.inspect_source_identity"),
            "remote_url": "https://github.com/elan6666/GraD-Pert.git",
            "remote_ref": preflight.SOURCE_REMOTE_REF,
            "published_commit": commit,
            "formal_eligible": True,
            "formal_eligibility_reason": None,
            "publication_verification_method": "hash_pinned_source_publication_receipt",
            "publication_receipt_path": str(publication.resolve()),
            "publication_receipt_sha256": publication_sha,
            "publication_receipt_size_bytes": publication.stat().st_size,
            "source_dirty": False,
        },
        load_data_identity=lambda _root, _protocol: _data_identity(preflight),
        load_graph=load_graph,
        verify_artifact=lambda path, sha, size: {
            "path": str(path),
            "sha256": sha,
            "size_bytes": size,
        },
    )


def _run(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
    *,
    missing_hvg: int | None = None,
    bad_hvg: int | None = None,
    tampered_source_hvg: int | None = None,
    escaping_source_hvg: int | None = None,
    wrong_genept: bool = False,
    genept_selected_count_delta: int = 0,
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    loaded, roots = _write_graph_slots(
        preflight,
        data_root,
        matrix_sha256,
        missing_hvg=missing_hvg,
        tampered_source_hvg=tampered_source_hvg,
        escaping_source_hvg=escaping_source_hvg,
    )
    h512 = _manifest(512)
    genept_path = _genept_receipt(
        preflight,
        data_root=data_root,
        graph_relative=roots[512],
        manifest=h512,
        wrong_topology=wrong_genept,
        selected_count_delta=genept_selected_count_delta,
    )
    publication_receipt = tmp_path / "source-publication-receipt.json"
    publication_receipt.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    publication_receipt_sha256 = hashlib.sha256(publication_receipt.read_bytes()).hexdigest()
    return preflight.build_preflight_receipt(
        matrix_path=MATRIX,
        expected_matrix_sha256=matrix_sha256,
        repository_root=PROJECT_ROOT,
        expected_source_commit=SOURCE_COMMIT,
        source_publication_receipt=publication_receipt,
        source_publication_receipt_sha256=publication_receipt_sha256,
        data_root=data_root,
        genept_preflight_receipt=genept_path,
        genept_preflight_receipt_sha256=hashlib.sha256(genept_path.read_bytes()).hexdigest(),
        dependencies=_dependencies(preflight, loaded, bad_hvg=bad_hvg),
    )


def test_all_25_rows_close_when_all_graphs_and_genept_are_ready(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    receipt = _run(preflight, tmp_path, matrix_sha256)
    assert receipt["status"] == "passed"
    assert receipt["matrix_row_count"] == 25
    assert len(receipt["rows"]) == 25
    assert receipt["row_status_counts"] == {"passed": 25, "blocked": 0}
    assert receipt["cross_h_audit"]["status"] == "passed"
    assert receipt["scientific_completion"] is False
    assert receipt["source"]["source_tree_sha256"] == "d" * 64
    assert receipt["source"]["git_tree_object"] == "c" * 40
    assert receipt["runtime_import_guard"]["status"] == "passed"
    assert receipt["runtime_import_guard"]["before"]["torch_loaded"] is False
    assert receipt["runtime_import_guard"]["after"]["torch_cuda_loaded"] is False
    assert receipt["runtime_import_guard"]["new_torch_modules"] == []
    assert receipt["forbidden_runtime"] == {
        "cuda_initialized": False,
        "model_constructed": False,
        "canonical_training_data_constructed": False,
        "canonical_validation_data_constructed": False,
        "canonical_test_data_constructed": False,
    }
    by_id = {row["variant_id"]: row for row in receipt["rows"]}
    baseline = by_id["a0_ratio_ring_half"]
    assert baseline["binding"]["config_sha256"]
    assert baseline["architecture"]["local_view_builder"] == "ring_induced"
    assert baseline["data_binding"]["dataset_id"] == "nadig_jurkat"
    assert baseline["data_binding"]["protocol_id"] == "within_cell_unseen_single"
    assert baseline["data_binding"]["canonical_manifest_sha256"] == "1" * 64
    assert baseline["data_binding"]["canonical_data_sha256"] == "2" * 64
    assert baseline["data_binding"]["split_manifest_sha256"] == "3" * 64
    assert baseline["data_binding"]["split_content_sha256"] == "4" * 64
    assert baseline["data_binding"]["source_manifest_sha256"] == "5" * 64
    assert baseline["data_binding"]["source_h5ad_sha256"] == "6" * 64
    assert set(baseline["data_binding"]["artifacts"]) == {
        "canonical_manifest",
        "canonical_h5ad",
        "split_manifest",
        "source_manifest",
        "source_h5ad",
    }
    assert all(
        Path(artifact["path"]).is_absolute()
        and len(artifact["sha256"]) == 64
        and artifact["size_bytes"] > 0
        and artifact["role"]
        for artifact in baseline["data_binding"]["artifacts"].values()
    )
    assert Path(baseline["runtime_graph_root_path"]).is_absolute()
    assert set(baseline["graph"]["artifacts"]) == {
        "manifest",
        "graph_gene_ids",
        "hvg_dispersion_ranking",
        "go",
        "string",
    }
    assert all(
        Path(artifact["path"]).is_absolute()
        and len(artifact["sha256"]) == 64
        and artifact["size_bytes"] > 0
        and artifact["role"]
        for artifact in baseline["graph"]["artifacts"].values()
    )
    assert baseline["graph"]["canonical_data_sha256"] == "2" * 64
    assert baseline["graph"]["split_content_sha256"] == "4" * 64
    assert baseline["graph"]["candidate_target_order_sha256"]
    assert by_id["a0_ratio_ring_half"]["local_view_contract"]["effective_node_budget"] == 261
    assert by_id["a0_ratio_ring_half"]["local_view_contract"]["node_budget_remainder"] == 0
    assert by_id["l3_ring_quarter"]["local_view_contract"]["effective_node_budget"] == 130
    assert by_id["l3_ring_quarter"]["local_view_contract"]["node_budget_remainder"] == 2
    assert by_id["l4_ring_half_mask_half"]["local_view_contract"]["effective_mask_view_count"] == 4
    assert all(row["anchor_capacity"]["checked"] is False for row in receipt["rows"])
    assert all(
        row["genept"]["status"] == "passed"
        for row in receipt["rows"]
        if row["variant_id"].startswith("e")
    )
    for row in receipt["rows"]:
        if not row["variant_id"].startswith("e"):
            continue
        genept_receipt = row["genept"]["receipt"]
        assert set(genept_receipt) == {"path", "sha256", "size_bytes"}
        assert genept_receipt["size_bytes"] == Path(genept_receipt["path"]).stat().st_size


def test_missing_h_graph_is_blocked_not_available(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    receipt = _run(preflight, tmp_path, matrix_sha256, missing_hvg=2048)
    assert receipt["status"] == "blocked"
    h2 = next(row for row in receipt["rows"] if row["variant_id"] == "h2_hvg2048_ratio_half")
    assert h2["status"] == "blocked_missing_graph"
    assert h2["graph"] is None
    assert receipt["cross_h_audit"]["status"] == "blocked_missing_graph"


def test_wrong_graph_artifact_hash_blocks_affected_row(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    receipt = _run(preflight, tmp_path, matrix_sha256, bad_hvg=1024)
    h1 = next(row for row in receipt["rows"] if row["variant_id"] == "h1_hvg1024_ratio_half")
    assert h1["status"] == "blocked_invalid_graph"
    assert "artifact hash mismatch" in h1["reasons"][0]
    assert receipt["status"] == "blocked"


def test_independent_source_artifact_rehash_blocks_tampering(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    receipt = _run(
        preflight,
        tmp_path,
        matrix_sha256,
        tampered_source_hvg=1024,
    )
    affected = [row for row in receipt["rows"] if row["architecture"]["graph_hvg_count"] == 1024]
    assert affected
    assert all(row["status"] == "blocked_invalid_graph" for row in affected)
    assert all("artifact SHA-256 differs: go" in row["reasons"][0] for row in affected)


def test_runtime_graph_artifact_symlink_cannot_escape_data_root(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    receipt = _run(
        preflight,
        tmp_path,
        matrix_sha256,
        escaping_source_hvg=2048,
    )
    affected = [row for row in receipt["rows"] if row["architecture"]["graph_hvg_count"] == 2048]
    assert affected
    assert all(row["status"] == "blocked_invalid_graph" for row in affected)
    assert all("escapes its frozen root" in row["reasons"][0] for row in affected)


def test_genept_receipt_identity_mismatch_blocks_exact_four_e_rows(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    receipt = _run(preflight, tmp_path, matrix_sha256, wrong_genept=True)
    blocked = [row for row in receipt["rows"] if row["status"] != "passed"]
    assert {row["variant_id"] for row in blocked} == {
        "e1_frozen_genept",
        "e2_genept_id_residual",
        "e3_genept_initialized",
        "es_genept_shuffle",
    }
    assert all(row["genept"]["status"] == "blocked_identity_mismatch" for row in blocked)


def test_genept_selected_axis_count_mismatch_blocks_exact_four_e_rows(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    receipt = _run(
        preflight,
        tmp_path,
        matrix_sha256,
        genept_selected_count_delta=1,
    )
    blocked = [row for row in receipt["rows"] if row["status"] != "passed"]
    assert {row["variant_id"] for row in blocked} == {
        "e1_frozen_genept",
        "e2_genept_id_residual",
        "e3_genept_initialized",
        "es_genept_shuffle",
    }
    assert all("selected_gene_count" in row["genept"]["reason"] for row in blocked)


def test_matrix_tampering_fails_before_any_row_is_claimed(
    preflight: ModuleType,
    tmp_path: Path,
    matrix_sha256: str,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(ValueError, match="matrix SHA-256 differs"):
        preflight.build_preflight_receipt(
            matrix_path=MATRIX,
            expected_matrix_sha256="0" * 64,
            repository_root=PROJECT_ROOT,
            expected_source_commit=SOURCE_COMMIT,
            source_publication_receipt=tmp_path / "not-inspected-publication.json",
            source_publication_receipt_sha256="f" * 64,
            data_root=data_root,
            genept_preflight_receipt=None,
            genept_preflight_receipt_sha256=None,
            dependencies=preflight.PreflightDependencies(
                inspect_source=lambda root, commit, publication, publication_sha: {},
                load_data_identity=lambda root, protocol: _data_identity(preflight),
                load_graph=lambda root: (None, None),
                verify_artifact=lambda path, sha, size: {},
            ),
        )
    assert matrix_sha256 != "0" * 64


def test_data_identity_rejects_relative_rehash_path(preflight: ModuleType) -> None:
    valid = _data_identity(preflight)
    invalid = preflight.DataIdentity(
        **{
            **valid.__dict__,
            "canonical_data_path": "relative/canonical.h5ad",
        }
    )
    with pytest.raises(preflight.PreflightError, match="path must be absolute"):
        preflight._validate_data_identity_evidence(
            invalid,
            protocol_id="within_cell_unseen_single",
        )


def test_runtime_import_guard_is_measured_and_fail_closed_in_fresh_process() -> None:
    code = f"""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

script = Path({str(SCRIPT)!r})
spec = importlib.util.spec_from_file_location("fresh_p0_import_guard", script)
if spec is None or spec.loader is None:
    raise RuntimeError("failed to load fresh P0 module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
clean = module._require_no_torch_imports("fresh-process-entry")
sys.modules["torch"] = ModuleType("torch")
try:
    module._require_no_torch_imports("fresh-process-injected")
except module.PreflightError as error:
    failure = str(error)
else:
    raise AssertionError("P0 import guard accepted an injected Torch module")
print(json.dumps({{"clean": clean, "failure": failure}}, sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["clean"]["measurement_method"] == (
        "sys.modules_snapshot_without_importing_torch"
    )
    assert payload["clean"]["torch_loaded"] is False
    assert payload["clean"]["torch_cuda_loaded"] is False
    assert "loaded Torch/CUDA module" in payload["failure"]


def test_source_inspection_binds_git_and_exact_content_tree(
    preflight: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    (repository / "src/package").mkdir(parents=True)
    (repository / "src/package/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repository / "AGENTS.md").write_text("fixture\n", encoding="utf-8")
    commands = (
        ("init",),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test User"),
        ("remote", "add", "origin", "https://github.com/elan6666/GraD-Pert.git"),
        ("add", "."),
        ("commit", "-m", "fixture"),
    )
    for arguments in commands:
        subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    publication_receipt = tmp_path / "source-publication-receipt.json"
    publication_receipt.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    publication_receipt_sha256 = hashlib.sha256(publication_receipt.read_bytes()).hexdigest()
    observed_calls: list[dict[str, object]] = []

    def inspect_formal_source(root: Path, **kwargs: object) -> SimpleNamespace:
        observed_calls.append(dict(kwargs))
        observed_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return SimpleNamespace(
            commit=observed_commit,
            dirty=dirty,
            tree_sha256="d" * 64,
            remote_url="https://github.com/elan6666/GraD-Pert.git",
            remote_ref=preflight.SOURCE_REMOTE_REF,
            published_commit=observed_commit,
            formal_eligible=True,
            formal_eligibility_reason=None,
            publication_receipt_sha256=publication_receipt_sha256,
        )

    monkeypatch.setattr(preflight, "inspect_source_identity", inspect_formal_source)
    evidence = preflight._inspect_clean_source(
        repository,
        commit,
        publication_receipt,
        publication_receipt_sha256,
    )
    assert evidence["observed_commit"] == commit
    assert len(evidence["git_tree_object"]) == 40
    assert len(evidence["source_tree_sha256"]) == 64
    assert evidence["source_dirty"] is False
    assert evidence["published_commit"] == commit
    assert evidence["formal_eligible"] is True
    assert observed_calls == [
        {
            "formal": True,
            "expected_repository": preflight.EXPECTED_REPOSITORY,
            "remote_ref": preflight.SOURCE_REMOTE_REF,
            "publication_receipt": publication_receipt.resolve(),
            "expected_publication_receipt_sha256": publication_receipt_sha256,
        }
    ]

    (repository / "src/package/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="clean source worktree"):
        preflight._inspect_clean_source(
            repository,
            commit,
            publication_receipt,
            publication_receipt_sha256,
        )


def test_source_evidence_rejects_wrong_remote_and_unpublished_commit(
    preflight: ModuleType,
    tmp_path: Path,
) -> None:
    publication_receipt = tmp_path / "source-publication-receipt.json"
    publication_receipt.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    publication_receipt_sha256 = hashlib.sha256(publication_receipt.read_bytes()).hexdigest()
    source = _dependencies(preflight, {}).inspect_source(
        PROJECT_ROOT,
        SOURCE_COMMIT,
        publication_receipt,
        publication_receipt_sha256,
    )
    wrong_remote = {**source, "remote_url": "https://github.com/example/not-gradpert"}
    with pytest.raises(preflight.PreflightError, match="source remote differs"):
        preflight._validate_source_evidence(
            wrong_remote,
            repository_root=PROJECT_ROOT,
            expected_commit=SOURCE_COMMIT,
            publication_receipt=publication_receipt,
            expected_publication_receipt_sha256=publication_receipt_sha256,
        )

    unpublished = {
        **source,
        "published_commit": None,
        "formal_eligible": False,
        "formal_eligibility_reason": "not published",
    }
    with pytest.raises(preflight.PreflightError, match="source identity evidence differs"):
        preflight._validate_source_evidence(
            unpublished,
            repository_root=PROJECT_ROOT,
            expected_commit=SOURCE_COMMIT,
            publication_receipt=publication_receipt,
            expected_publication_receipt_sha256=publication_receipt_sha256,
        )


def test_source_publication_receipt_is_required_and_hash_pinned(
    preflight: ModuleType,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-publication.json"
    with pytest.raises(preflight.PreflightError, match="must be a regular file"):
        preflight._inspect_clean_source(
            PROJECT_ROOT,
            SOURCE_COMMIT,
            missing,
            "f" * 64,
        )

    publication_receipt = tmp_path / "source-publication-receipt.json"
    publication_receipt.write_text("fixture\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="SHA-256 differs"):
        preflight._inspect_clean_source(
            PROJECT_ROOT,
            SOURCE_COMMIT,
            publication_receipt,
            "f" * 64,
        )


def test_cli_failure_still_writes_atomic_non_scientific_receipt(
    preflight: ModuleType,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "failed.json"
    publication_receipt = tmp_path / "source-publication-receipt.json"
    publication_receipt.write_text("fixture\n", encoding="utf-8")
    publication_receipt_sha256 = hashlib.sha256(publication_receipt.read_bytes()).hexdigest()
    argv = [
        "--matrix",
        str(MATRIX),
        "--expected-matrix-sha256",
        "0" * 64,
        "--repository-root",
        str(PROJECT_ROOT),
        "--expected-source-commit",
        SOURCE_COMMIT,
        "--source-publication-receipt",
        str(publication_receipt),
        "--source-publication-receipt-sha256",
        publication_receipt_sha256,
        "--data-root",
        str(tmp_path),
        "--receipt",
        str(destination),
    ]
    result = preflight.main(argv)
    assert result == 1
    sealed_bytes = destination.read_bytes()
    payload = json.loads(sealed_bytes)
    assert payload["status"] == "failed"
    assert payload["scientific_completion"] is False
    assert payload["primary_failure"]["type"] in {"PreflightError", "ValueError"}

    duplicate_result = preflight.main(argv)
    assert duplicate_result == 1
    assert destination.read_bytes() == sealed_bytes


def test_cli_rejects_existing_symlink_and_directory_before_inspection(
    preflight: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_inspection(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("existing receipt must fail before P0 inspection")

    monkeypatch.setattr(preflight, "build_preflight_receipt", reject_inspection)
    publication_receipt = tmp_path / "source-publication-receipt.json"
    publication_receipt.write_text("fixture\n", encoding="utf-8")
    publication_receipt_sha256 = hashlib.sha256(publication_receipt.read_bytes()).hexdigest()
    directory = tmp_path / "existing-directory"
    directory.mkdir()
    symlink = tmp_path / "existing-symlink"
    symlink.symlink_to(tmp_path / "missing-target")
    for destination in (directory, symlink):
        result = preflight.main(
            [
                "--matrix",
                str(MATRIX),
                "--expected-matrix-sha256",
                "0" * 64,
                "--repository-root",
                str(PROJECT_ROOT),
                "--expected-source-commit",
                SOURCE_COMMIT,
                "--source-publication-receipt",
                str(publication_receipt),
                "--source-publication-receipt-sha256",
                publication_receipt_sha256,
                "--data-root",
                str(tmp_path),
                "--receipt",
                str(destination),
            ]
        )
        assert result == 1
    assert directory.is_dir()
    assert symlink.is_symlink()
    assert symlink.readlink() == tmp_path / "missing-target"


def test_p0_receipt_claim_is_atomic_and_preserves_first_owner(
    preflight: ModuleType,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "p0.json"
    preflight._claim_json_output(destination, {"status": "claimed", "owner": "first"})
    sealed = destination.read_bytes()
    with pytest.raises(FileExistsError):
        preflight._claim_json_output(destination, {"status": "claimed", "owner": "second"})
    assert destination.read_bytes() == sealed
