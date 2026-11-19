"""Frozen identities shared by the external perturbation holdout study."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER = Path("/data/yilangliu/GraD-Pert")
STUDY = SERVER / "development/perturbation-holdout-study-20261117-a1"
CROSS = SERVER / "crosscell-fixed-v1"
DATA = SERVER / "data-vnext-a942114"
DATASET = "nadig_jurkat"
PROTOCOL = "within_cell_unseen_single"
COHORT_SHA = "543a5b5638030da5df5790423ad7efec636e298e12089b5d98b534688cb9d4f4"
TXPERT_CHECKOUT = SERVER / "upstreams/txpert"
TXPERT_COMMIT = "08d82eea86746b044cf7531f4ec8c5f60e1cb73f"
