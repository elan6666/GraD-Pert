"""Public TxPert training recipes recovered from the authors' released states."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleasedRecipe:
    config_file: str
    checkpoint_member: str
    checkpoint_sha256: str
    epochs: int
    learning_rate: float
    patience: int
    checkpoint_every_n_epochs: int
    warmup_epochs: int = 5
    weight_decay: float = 0.0
    min_lr: float = 0.0

    @property
    def scheduler_args(self) -> dict[str, str | int]:
        return {
            "type": "cosine",
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.epochs,
        }


ZENODO_RECORD = "https://zenodo.org/records/15420279"

CROSS_CELL_GAT = ReleasedRecipe(
    config_file="configs/config-x-cell-gat.yaml",
    checkpoint_member="checkpoints/K562_unseen_cell_gat.ckpt",
    checkpoint_sha256="22994e5097255f75fc4196bddea8e8227dd4c110694a7794f066e041fbf42763",
    epochs=80,
    learning_rate=0.0003,
    patience=100,
    checkpoint_every_n_epochs=5,
)

WITHIN_CELL_EXPHORMER_MG = ReleasedRecipe(
    config_file="configs/config-exphormer-mg.yaml",
    checkpoint_member="checkpoints/K562_unseen_pert_exphormer_mg.ckpt",
    checkpoint_sha256="63df6c18b150e918f359b666c82de9ad9826db8e5f1ed771839d18e5dbd6a11c",
    epochs=50,
    learning_rate=0.001,
    patience=40,
    checkpoint_every_n_epochs=10,
)
