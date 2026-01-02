# indicators/indicator_base.py
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class IndicatorBase(ABC):
    """Common interface for technical indicators."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def calculate(self) -> Any:
        """Return indicator values without mutating input frame."""
        raise NotImplementedError
