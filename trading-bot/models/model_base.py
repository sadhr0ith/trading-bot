# models/model_base.py
from abc import ABC, abstractmethod
from utils.logger import setup_logger

class ModelBase(ABC):
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def log_action(self, message, level="info"):
        getattr(self.logger, level.lower())(message)
