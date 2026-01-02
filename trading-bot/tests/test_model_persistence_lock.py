import os
import time
from pathlib import Path

from trading_bot.utils.model_persistence import ModelPersistence


def test_stale_lock_is_removed_on_load(tmp_path):
    mp = ModelPersistence(base_dir=tmp_path / "models")
    mp.STALE_LOCK_TTL_SECONDS = 0.1  # speed up for test
    strategy = "test_strategy"
    lock_path = mp._strategy_dir(strategy) / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("locked", encoding="utf-8")
    stale_mtime = time.time() - 5  # sufficiently old
    os.utime(lock_path, (stale_mtime, stale_mtime))

    # load should detect stale lock and remove it without raising
    artifact = mp.load(strategy)
    assert artifact is None
    assert not lock_path.exists()


def test_load_metadata_without_model(tmp_path):
    mp = ModelPersistence(base_dir=tmp_path / "models")
    strategy = "meta_only"
    # Save minimal artifact to create metadata
    mp.save(strategy, model="dummy", scaler=None, metadata={"feature_columns": ["Close"]})
    meta = mp.load_metadata(strategy)
    assert meta is not None
    assert meta.get("strategy") == strategy
    assert "saved_at" in meta
