from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from utils.model_persistence import ModelPersistence


def train_or_load_pipeline(
    key: str,
    persistence: ModelPersistence,
    pipeline_factory: Callable[[], Pipeline],
    X,
    y,
    feature_columns: Iterable[str],
    min_splits: int = 2,
    max_splits: int = 5,
    split_divisor: int = 50,
    scoring: str = "neg_mean_absolute_error",
    metadata: dict | None = None,
    config_signature_data: dict[str, Any] | None = None,
    tuner: Callable[[Pipeline, Any, Any, TimeSeriesSplit, str], tuple[Pipeline, float | None, dict[str, Any]]] | None = None,
) -> tuple[Pipeline, float | None]:
    """
    Load pipeline artifact if up-to-date, otherwise train with TimeSeriesSplit CV and persist.
    Returns (pipeline, mean_cv_metric).
    """
    latest_idx = X.index.max() if hasattr(X, "index") else None
    pipeline = pipeline_factory()
    config_signature = (
        _build_config_signature(config_signature_data) if config_signature_data else _build_model_signature(pipeline)
    )

    artifact = persistence.load(key)
    if artifact:
        meta = artifact.get("metadata", {})
        if (
            meta.get("trained_until") == str(latest_idx)
            and meta.get("feature_columns") == list(feature_columns)
            and meta.get("config_signature") == config_signature
        ):
            return artifact.get("model"), meta.get("mae_cv")

    n_splits = min(max_splits, max(min_splits, len(X) // split_divisor))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    tuner_meta: dict[str, Any] = {}
    if tuner:
        pipeline, cv_metric, tuner_meta = tuner(pipeline, X, y, tscv, scoring)
    else:
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring=scoring)
        cv_metric = (
            float(np.mean(np.abs(cv_scores))) if "neg_mean_absolute_error" in scoring else float(np.mean(cv_scores))
        )
        pipeline.fit(X, y)

    model_signature = _build_model_signature(pipeline)
    meta = {
        "trained_until": str(latest_idx),
        "feature_columns": list(feature_columns),
        "mae_cv": cv_metric if "neg_mean_absolute_error" in scoring else None,
        "model_signature": model_signature,
        "config_signature": config_signature,
        "scoring": scoring,
        "n_splits": n_splits,
    }
    if metadata:
        meta.update(metadata)
    if tuner_meta:
        meta.update(tuner_meta)
    persistence.save(key, pipeline, scaler=None, metadata=meta)
    return pipeline, cv_metric


def _build_model_signature(pipeline: Pipeline) -> str:
    """Return a stable signature for the model step to guard persistence reuse."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return "unknown"
    params = model.get_params(deep=False)
    try:
        params_blob = json.dumps(params, default=str, sort_keys=True)
    except TypeError:
        params_blob = json.dumps({k: str(v) for k, v in params.items()}, sort_keys=True)
    digest = hashlib.md5(params_blob.encode("utf-8")).hexdigest()
    return f"{model.__class__.__name__}:{digest}"


def _build_config_signature(config: dict[str, Any]) -> str:
    """Hash arbitrary config dict for persistence reuse."""
    try:
        blob = json.dumps(config, sort_keys=True, default=str)
    except TypeError:
        blob = json.dumps({k: str(v) for k, v in config.items()}, sort_keys=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()
