from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
        self.log_action("Initialized Random Forest model", "info")

    def log_action(self, message, level):
        # Placeholder for actual logging functionality
        print(f"{level.upper()}: {message}")

    def train(self, X_train, y_train):
        self.log_action("Training Random Forest model...", "info")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        self.log_action("Making predictions with Random Forest model...", "info")
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        self.log_action(f"Mean Absolute Error (MAE) on test set: {mae:.4f}", "info")
        return mae
