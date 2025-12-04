from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PersistenceMetadata(BaseModel):
    saved_at: datetime = Field(default_factory=datetime.utcnow)
    version: str
    strategy: str
    artifact_path: str
    feature_columns: list[str] = Field(default_factory=list)
    trained_until: str | None = None
    mae_cv: float | None = None
    config_signature: str | None = None
    model_signature: str | None = None
    scoring: str | None = None
    n_splits: int | None = None

    class Config:
        extra = "allow"
