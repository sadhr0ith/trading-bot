import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from trading_bot.core.exceptions import ModelPersistenceError
from trading_bot.utils.model_persistence import ModelPersistence
from trading_bot.utils.strategy_helpers import train_or_load_pipeline


def test_train_or_load_pipeline_load_only_returns_persisted_model(tmp_path):
    persistence = ModelPersistence(base_dir=tmp_path / "models")
    pipeline = Pipeline([("model", DummyRegressor(strategy="mean"))])
    # Persist dummy model with minimal metadata
    persistence.save(
        "test_strategy",
        model=pipeline,
        scaler=None,
        metadata={"feature_columns": ["x"]},
    )

    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    loaded, cv = train_or_load_pipeline(
        key="test_strategy",
        persistence=persistence,
        pipeline_factory=lambda: pipeline,  # unused in load_only
        X=X,
        y=y,
        feature_columns=["x"],
        load_only=True,
    )

    assert loaded is not None
    assert cv is None or isinstance(cv, float)


def test_train_or_load_pipeline_load_only_raises_if_missing(tmp_path):
    persistence = ModelPersistence(base_dir=tmp_path / "models_missing")
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([1, 2])
    with pytest.raises(ModelPersistenceError):
        train_or_load_pipeline(
            key="missing_strategy",
            persistence=persistence,
            pipeline_factory=lambda: Pipeline([("model", DummyRegressor(strategy="mean"))]),
            X=X,
            y=y,
            feature_columns=["x"],
            load_only=True,
        )


def test_train_or_load_pipeline_gap_fallback_trains(tmp_path):
    persistence = ModelPersistence(base_dir=tmp_path / "models_gap")
    pipeline = Pipeline([("model", DummyRegressor(strategy="mean"))])
    X = pd.DataFrame({"x": list(range(10))})
    y = pd.Series(list(range(10)))

    trained, cv = train_or_load_pipeline(
        key="gap_strategy",
        persistence=persistence,
        pipeline_factory=lambda: pipeline,
        X=X,
        y=y,
        feature_columns=["x"],
        gap=9,
    )

    assert trained is not None
    assert cv is None or isinstance(cv, float)
