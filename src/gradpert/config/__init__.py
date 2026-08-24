"""Strict experiment configuration loading and matrix validation."""

from gradpert.config.loader import load_experiment_config
from gradpert.config.matrix import verify_config_matrix
from gradpert.config.schema import ExperimentConfig

__all__ = ["ExperimentConfig", "load_experiment_config", "verify_config_matrix"]
