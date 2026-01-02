from __future__ import annotations

import json
import pickle
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from filelock import FileLock, Timeout
from pydantic import ValidationError as PydanticValidationError

from trading_bot.core.exceptions import ModelPersistenceError
from trading_bot.models.persistence import PersistenceMetadata
from trading_bot.utils.logger import setup_logger


class ModelPersistence:
    LOCK_TIMEOUT_SECONDS = 10
    STALE_LOCK_TTL_SECONDS = 300

    def __init__(self, base_dir: Path | str = Path("saved_models")):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)

    def _strategy_dir(self, strategy: str) -> Path:
        return self.base_dir / strategy

    def _metadata_path(self, strategy: str, version: str | None = None) -> Path:
        if version:
            return self._strategy_dir(strategy) / f"{version}.json"
        return self.base_dir / f"{strategy}_metadata.json"

    def _artifact_path(self, strategy: str, version: str | None = None, suffix: str = "pkl") -> Path:
        if version:
            return self._strategy_dir(strategy) / f"{version}.{suffix}"
        return self.base_dir / f"{strategy}_model.{suffix}"

    def _scaler_path(self, strategy: str, version: str | None = None) -> Path:
        suffix = "scaler.pkl" if version else "scaler.pkl"
        if version:
            return self._strategy_dir(strategy) / f"{version}_{suffix}"
        return self.base_dir / f"{strategy}_{suffix}"

    def _latest_version(self, strategy: str) -> str | None:
        strategy_dir = self._strategy_dir(strategy)
        if not strategy_dir.exists():
            return None
        metadata_files = sorted(strategy_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not metadata_files:
            return None
        return metadata_files[0].stem

    def load_metadata(self, strategy: str, version: str | None = None) -> dict[str, Any] | None:
        """Load persisted metadata without loading the model artifact."""
        with self._lock(strategy):
            target_version = version or self._latest_version(strategy)
            meta_path = self._metadata_path(strategy, target_version)
            if not meta_path.exists():
                return None
            try:
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                    return PersistenceMetadata(**metadata).model_dump()
            except (json.JSONDecodeError, PydanticValidationError, OSError) as exc:
                self.logger.warning(f"Failed to load metadata for {strategy}: {exc}")
                return None

    @contextmanager
    def _lock(self, strategy: str):
        """File-based lock to prevent concurrent save/load corruption."""
        lock_path = self._strategy_dir(strategy) / ".lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if lock_path.exists():
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > self.STALE_LOCK_TTL_SECONDS:
                    self.logger.warning(
                        f"Stale persistence lock detected for {strategy}; removing after {age:.1f}s."
                    )
                    lock_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
        file_lock = FileLock(str(lock_path))
        try:
            file_lock.acquire(timeout=self.LOCK_TIMEOUT_SECONDS)
        except Timeout as exc:
            raise ModelPersistenceError(f"Timed out acquiring lock for {strategy} at {lock_path}") from exc
        try:
            yield
        finally:
            # Cleanup: silently ignore errors releasing lock (may already be released)
            try:
                if file_lock.is_locked:
                    file_lock.release()
            except (OSError, RuntimeError):
                pass
            # Remove lock file (may already be deleted by another process)
            try:
                lock_path.unlink()
            except OSError:
                pass

    def save(
        self,
        strategy: str,
        model: Any,
        scaler: Any,
        metadata: dict | None = None,
        version: str | None = None,
        is_keras: bool = False,
    ) -> None:
        version = version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        strategy_dir = self._strategy_dir(strategy)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        with self._lock(strategy):
            if is_keras:
                artifact_path = self._artifact_path(strategy, version, suffix="keras")
                scaler_path = self._scaler_path(strategy, version)
                model.save(artifact_path)
                joblib.dump({"scaler": scaler}, scaler_path)
            else:
                artifact_path = self._artifact_path(strategy, version)
                joblib.dump({"model": model, "scaler": scaler}, artifact_path)

            metadata = metadata or {}
            metadata.update(
                {
                    "saved_at": datetime.now(timezone.utc),
                    "version": version,
                    "strategy": strategy,
                    "artifact_path": str(artifact_path),
                }
            )
            try:
                metadata = PersistenceMetadata(**metadata).model_dump()
            except PydanticValidationError as exc:
                self.logger.error(f"Metadata validation failed: {exc}")
            with open(self._metadata_path(strategy, version), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.info(f"Saved model artifact for {strategy} to {artifact_path}")

    def _load_artifact(self, strategy: str, version: str | None, is_keras: bool) -> dict[str, Any] | None:
        if is_keras:
            artifact_path = self._artifact_path(strategy, version, suffix="keras")
            scaler_path = self._scaler_path(strategy, version)
        else:
            artifact_path = self._artifact_path(strategy, version)
            scaler_path = None

        meta_path = self._metadata_path(strategy, version)
        if not artifact_path.exists():
            return None

        metadata: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    metadata = json.load(f)
                    metadata = PersistenceMetadata(**metadata).model_dump()
            except (json.JSONDecodeError, PydanticValidationError, OSError) as exc:
                self.logger.warning(f"Failed to parse metadata for {strategy}: {exc}")
                metadata = {}

        if is_keras:
            try:
                from tensorflow import keras  # type: ignore
            except ImportError as exc:
                raise ModelPersistenceError(
                    "TensorFlow/Keras not available; cannot load persisted Keras model."
                ) from exc
            try:
                model = keras.models.load_model(artifact_path)
            except (OSError, ValueError, TypeError) as exc:
                # Keras can raise various errors: file issues, model format issues, weight issues
                raise ModelPersistenceError(f"Failed to load Keras model from {artifact_path}: {exc}") from exc
            scaler = None
            if scaler_path and scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path).get("scaler")
                except (OSError, pickle.UnpicklingError, ModuleNotFoundError) as exc:
                    self.logger.warning(f"Failed to load scaler for {strategy}: {exc}")
                    scaler = None
        else:
            try:
                artifacts = joblib.load(artifact_path)
            except ModuleNotFoundError as exc:
                self.logger.warning(f"Skipping corrupted artifact {artifact_path}: {exc}. Will retrain.")
                return None
            except (OSError, pickle.UnpicklingError, ValueError) as exc:
                self.logger.warning(f"Failed to load artifact {artifact_path}: {exc}")
                return None
            model = artifacts.get("model")
            scaler = artifacts.get("scaler")

        if version and "version" not in metadata:
            metadata["version"] = version
        return {"model": model, "scaler": scaler, "metadata": metadata}

    def load(self, strategy: str, version: str | None = None, is_keras: bool = False) -> dict[str, Any] | None:
        with self._lock(strategy):
            if version:
                return self._load_artifact(strategy, version, is_keras)

            latest_version = self._latest_version(strategy)
            artifact = self._load_artifact(strategy, latest_version, is_keras) if latest_version else None
            if artifact:
                return artifact

            # Legacy single-file fallback
            return self._load_artifact(strategy, None, is_keras)
