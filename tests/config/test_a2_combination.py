from pathlib import Path

from gradpert.config import NativeArchitectureOptions, load_experiment_config


def test_a2_is_exact_a1_plus_d5():
    root = Path(__file__).resolve().parents[2] / "configs/combinations"
    a1 = load_experiment_config(root / "a1_e3_l1_m1/gradpert_b2/nadig_jurkat.yaml")
    a2 = load_experiment_config(root / "a2_e3_l1_m1_d5/gradpert_b2/nadig_jurkat.yaml")
    p1 = {k: v.value for k, v in a1.model.parameters.items()}
    p2 = {k: v.value for k, v in a2.model.parameters.items()}
    assert p1.keys() == p2.keys()
    assert {k for k in p1 if p1[k] != p2[k]} == {"decoder_mode", "graph_tower_output_dim"}
    arch = NativeArchitectureOptions.from_parameters(a2.model.parameters)
    assert arch.decoder_mode == "concat"
    assert arch.graph_output_dim == 256
    assert p2["latent_dim"] == 64
    assert a1.training == a2.training
    assert a1.data == a2.data
    assert a1.evaluation == a2.evaluation
