import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error

class XGBoostModel:
    def __init__(self):
        """
        Initializes the XGBoost model.
        """
        self.model = None
        self.log_action("Initialized XGBoost model", "info")

    def log_action(self, message, level):
        # Placeholder logging function. Replace with actual logging mechanism.
        print(f"{level.upper()}: {message}")

    def train(self, X_train, y_train):
        """
        Trains the XGBoost model on the given training data.
        :param X_train: Feature matrix for training (2D array).
        :param y_train: Labels for training (1D array).
        """
        self.log_action("Training XGBoost model...", "info")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'reg:squarederror',  # This objective is for regression
            'eval_metric': 'rmse',  # Root Mean Squared Error as the evaluation metric
            'max_depth': 6,
            'eta': 0.1
        }
        num_round = 100  # Number of boosting rounds
        self.model = xgb.train(params, dtrain, num_round)

    def predict(self, X_test):
        """
        Makes predictions on the given test data using the trained model.
        :param X_test: Feature matrix for testing (2D array).
        :return: Predictions (1D array).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.log_action("Predicting with XGBoost model", "info")
        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model using Mean Absolute Error (MAE) on the test set.
        :param X_test: Feature matrix for testing (2D array).
        :param y_test: True labels for testing (1D array).
        :return: Mean Absolute Error (MAE).
        """
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        self.log_action(f"Mean Absolute Error (MAE) on test set: {mae:.4f}", "info")
        return mae
