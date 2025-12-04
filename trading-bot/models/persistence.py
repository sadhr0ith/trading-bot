from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PersistenceMetadata(BaseModel):
    saved_at: datetime = Field(default_factory=datetime.utcnow)
    version: str
    strategy: str
    artifact_path: str
    feature_columns: List[str] = Field(default_factory=list)
    trained_until: Optional[str] = None
    mae_cv: Optional[float] = None
    config_signature: Optional[str] = None
    model_signature: Optional[str] = None
    scoring: Optional[str] = None
    n_splits: Optional[int] = None

    class Config:
        extra = "allow"
