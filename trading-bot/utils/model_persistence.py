from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from pydantic import ValidationError as PydanticValidationError

from core.exceptions import ModelPersistenceError
from models.persistence import PersistenceMetadata
from utils.logger import setup_logger


class ModelPersistence:
    def __init__(self, base_dir: Path | str = Path("saved_models")):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)

    def _strategy_dir(self, strategy: str) -> Path:
        return self.base_dir / strategy

    def _metadata_path(self, strategy: str, version: Optional[str] = None) -> Path:
        if version:
            return self._strategy_dir(strategy) / f"{version}.json"
        return self.base_dir / f"{strategy}_metadata.json"

    def _artifact_path(self, strategy: str, version: Optional[str] = None, suffix: str = "pkl") -> Path:
        if version:
            return self._strategy_dir(strategy) / f"{version}.{suffix}"
        return self.base_dir / f"{strategy}_model.{suffix}"

    def _scaler_path(self, strategy: str, version: Optional[str] = None) -> Path:
        suffix = "scaler.pkl" if version else "scaler.pkl"
        if version:
            return self._strategy_dir(strategy) / f"{version}_{suffix}"
        return self.base_dir / f"{strategy}_{suffix}"

    def _latest_version(self, strategy: str) -> Optional[str]:
        strategy_dir = self._strategy_dir(strategy)
        if not strategy_dir.exists():
            return None
        metadata_files = sorted(strategy_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not metadata_files:
            return None
        return metadata_files[0].stem

    def save(
        self,
        strategy: str,
        model: Any,
        scaler: Any,
        metadata: Optional[Dict] = None,
        version: Optional[str] = None,
        is_keras: bool = False,
    ) -> None:
        version = version or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        strategy_dir = self._strategy_dir(strategy)
        strategy_dir.mkdir(parents=True, exist_ok=True)

        if is_keras:
            artifact_path = self._artifact_path(strategy, version, suffix="keras")
            scaler_path = self._scaler_path(strategy, version)
            model.save(artifact_path)
            joblib.dump({"scaler": scaler}, scaler_path)
        else:
            artifact_path = self._artifact_path(strategy, version)
            joblib.dump({"model": model, "scaler": scaler}, artifact_path)

        metadata = metadata or {}
        metadata.update({
            "saved_at": datetime.utcnow(),
            "version": version,
            "strategy": strategy,
            "artifact_path": str(artifact_path),
        })
        try:
            metadata = PersistenceMetadata(**metadata).dict()
        except PydanticValidationError as exc:
            self.logger.error(f"Metadata validation failed: {exc}")
        with open(self._metadata_path(strategy, version), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Saved model artifact for {strategy} to {artifact_path}")

    def _load_artifact(self, strategy: str, version: Optional[str], is_keras: bool) -> Optional[Dict[str, Any]]:
        if is_keras:
            artifact_path = self._artifact_path(strategy, version, suffix="keras")
            scaler_path = self._scaler_path(strategy, version)
        else:
            artifact_path = self._artifact_path(strategy, version)
            scaler_path = None

        meta_path = self._metadata_path(strategy, version)
        if not artifact_path.exists():
            return None

        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    metadata = PersistenceMetadata(**metadata).dict()
            except Exception as exc:  # noqa: BLE001
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
            except Exception as exc:  # noqa: BLE001
                raise ModelPersistenceError(f"Failed to load Keras model from {artifact_path}: {exc}") from exc
            scaler = None
            if scaler_path and scaler_path.exists():
                scaler = joblib.load(scaler_path).get("scaler")
        else:
            artifacts = joblib.load(artifact_path)
            model = artifacts.get("model")
            scaler = artifacts.get("scaler")

        if version and "version" not in metadata:
            metadata["version"] = version
        return {"model": model, "scaler": scaler, "metadata": metadata}

    def load(self, strategy: str, version: Optional[str] = None, is_keras: bool = False) -> Optional[Dict[str, Any]]:
        if version:
            return self._load_artifact(strategy, version, is_keras)

        latest_version = self._latest_version(strategy)
        artifact = self._load_artifact(strategy, latest_version, is_keras) if latest_version else None
        if artifact:
            return artifact

        # Legacy single-file fallback
        return self._load_artifact(strategy, None, is_keras)
