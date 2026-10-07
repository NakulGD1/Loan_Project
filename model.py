"""Backward-compatible imports for the shared loan model implementation."""

from loan_model import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DEFAULT_THRESHOLD,
    FEATURE_SELECTION_COUNT,
    FEATURES,
    IMPORTANT_FEATURES,
    MODEL_PATH,
    MODEL_PATHS,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
    load_data,
    load_model,
    predict_one,
    train_model,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "DATA_PATH",
    "DEFAULT_THRESHOLD",
    "FEATURE_SELECTION_COUNT",
    "FEATURES",
    "IMPORTANT_FEATURES",
    "MODEL_PATH",
    "MODEL_PATHS",
    "NUMERIC_FEATURES",
    "TARGET",
    "build_pipeline",
    "load_data",
    "load_model",
    "predict_one",
    "train_model",
]
