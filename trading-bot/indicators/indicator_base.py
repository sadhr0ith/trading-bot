# indicators/indicator_base.py
from abc import ABC, abstractmethod

class IndicatorBase(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def calculate(self):
        pass
