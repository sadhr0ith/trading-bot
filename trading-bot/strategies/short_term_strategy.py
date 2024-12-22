from .strategy_base import StrategyBase
from models.xgboost_model import XGBoostModel
from indicators.macd import MACD
from sklearn.model_selection import train_test_split
from utils.email_notifications import send_email
import numpy as np

class ShortTermStrategy(StrategyBase):
    def execute(self):
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]

        self.log_action(f"Executing short-term strategy for {asset} using XGBoost and MACD", "info")

        # Calculate MACD and other indicators
        if "macd" in self.config["indicators"]:
            self.log_action("Calculating MACD indicator...", "info")
            macd_indicator = MACD(self.data)

            # Drop any existing MACD columns before recalculating
            self.data = self.data.drop(columns=['MACD', 'Signal', 'MACD_Histogram'], errors='ignore')

            # Calculate and join the new MACD data
            macd_data = macd_indicator.calculate()
            self.data = self.data.join(macd_data[['MACD', 'Signal', 'MACD_Histogram']])

        # Log MACD and Signal values for debugging
        self.log_action(f"MACD and Signal values (last 5 rows):\n{self.data[['MACD', 'Signal']].tail()}", "debug")

        # Prepare data for XGBoost (using Close price and MACD as features)
        X = self.data[['Close', 'MACD']]
        y = self.data['Close']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Verify that X_train and y_train have the expected shapes
        assert len(X_train.shape) == 2, f"X_train should be 2D (samples, features), but got {X_train.shape}"
        assert len(y_train.shape) == 1, f"y_train should be 1D (samples), but got {y_train.shape}"

        # Log data shapes for debugging
        self.log_action(f"Shape of training data: X_train: {X_train.shape}, y_train: {y_train.shape}", "debug")
        self.log_action(f"Shape of test data: X_test: {X_test.shape}, y_test: {y_test.shape}", "debug")

        # Train the XGBoost model
        self.log_action("Training the XGBoost model...", "info")
        xgboost_model = XGBoostModel()
        xgboost_model.train(X_train, y_train)

        # Evaluate the model on the test set using MAE
        self.log_action("Evaluating the XGBoost model on the test set...", "info")
        mae = xgboost_model.evaluate(X_test, y_test)
        self.log_action(f"Mean Absolute Error (MAE) on test set: {mae:.4f}", "info")

        # Make predictions using the test set
        predictions = xgboost_model.predict(X_test)

        # Log predictions and actual values for debugging
        self.log_action(f"First 5 predictions:\n{predictions[:5]}", "debug")
        self.log_action(f"First 5 actual values (y_test):\n{y_test[:5].values}", "debug")

        # Get the last prediction and actual close price
        last_pred = predictions[-1]
        last_close = y_test.iloc[-1]

                # Get the MACD and Signal line for the last day
        last_macd = self.data['MACD'].iloc[-1]
        last_signal = self.data['Signal'].iloc[-1]

        # Calculate the difference between predicted and actual close price
        price_diff = abs(last_pred - last_close)
        msg = f"Predicted Price: {last_pred:.2f}, Actual Close: {last_close:.2f}, MACD: {last_macd:.2f}, Signal: {last_signal:.2f}"

        # Define a threshold for HOLD signal (e.g., if difference is < 0.5%)
        hold_threshold = 0.005 * last_close

        # Trading signal logic with MACD confirmation
        if price_diff < hold_threshold:
            decision = "HOLD"
            self.log_action(f"{msg} -> {decision} signal (price difference: {price_diff:.2f} < threshold: {hold_threshold:.2f})", "warning")
            send_email(f"{decision} Signal for {asset} using {strategy_name}",
                       f"{msg} -> {decision} signal", self.config['notification_email'])
        elif last_pred > last_close and last_macd > last_signal:
            decision = "BUY"
            self.log_action(f"{msg} -> {decision} signal (predicted price higher and MACD supports BUY)", "info")
            send_email(f"{decision} Signal for {asset} using {strategy_name}",
                       f"{msg} -> {decision} signal", self.config['notification_email'])
        elif last_pred < last_close and last_macd < last_signal:
            decision = "SELL"
            self.log_action(f"{msg} -> {decision} signal (predicted price lower and MACD supports SELL)", "info")
            send_email(f"{decision} Signal for {asset} using {strategy_name}",
                       f"{msg} -> {decision} signal", self.config['notification_email'])
        else:
            decision = "HOLD"
            self.log_action(f"{msg} -> {decision} signal (predicted price and MACD not aligned)", "warning")
            send_email(f"{decision} Signal for {asset} using {strategy_name}",
                       f"{msg} -> {decision} signal", self.config['notification_email'])
