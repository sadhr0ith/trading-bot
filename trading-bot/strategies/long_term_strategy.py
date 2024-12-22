from .strategy_base import StrategyBase
from models.random_forest_model import RandomForestModel
from indicators.sma import SMA  # Import SMA indicator (or implement it)
from indicators.ema import EMA  # Import EMA indicator (or implement it)
from indicators.macd import MACD
from sklearn.model_selection import train_test_split
from utils.email_notifications import send_email
import numpy as np

class LongTermStrategy(StrategyBase):
    def execute(self):
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]

        self.log_action(f"Executing long-term strategy for {asset} using Random Forest and SMA/EMA", "info")

        # Calculate long-term indicators like SMA, EMA, and possibly MACD
        if "sma" in self.config["indicators"]:
            self.log_action("Calculating SMA indicator...", "info")
            sma_indicator = SMA(self.data)
            self.data = self.data.join(sma_indicator.calculate())

        if "ema" in self.config["indicators"]:
            self.log_action("Calculating EMA indicator...", "info")
            ema_indicator = EMA(self.data)
            self.data = self.data.join(ema_indicator.calculate())

        if "macd" in self.config["indicators"]:
            self.log_action("Calculating MACD indicator...", "info")
            macd_indicator = MACD(self.data)
            self.data = self.data.join(macd_indicator.calculate())

        # Log indicator values for debugging
        self.log_action(f"Indicators (last 5 rows):\n{self.data[['SMA', 'EMA', 'MACD']].tail()}", "debug")

        # Prepare data for Random Forest (using Close price, SMA, EMA, and MACD as features)
        X = self.data[['Close', 'SMA', 'EMA', 'MACD']]
        y = self.data['Close']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Verify shapes
        assert len(X_train.shape) == 2, f"X_train should be 2D (samples, features), but got {X_train.shape}"
        assert len(y_train.shape) == 1, f"y_train should be 1D (samples), but got {y_train.shape}"

        # Log data shapes for debugging
        self.log_action(f"Shape of training data: X_train: {X_train.shape}, y_train: {y_train.shape}", "debug")
        self.log_action(f"Shape of test data: X_test: {X_test.shape}, y_test: {y_test.shape}", "debug")

        # Train the Random Forest model
        self.log_action("Training the Random Forest model...", "info")
        random_forest_model = RandomForestModel()
        random_forest_model.train(X_train, y_train)

        # Make predictions using the test set
        predictions = random_forest_model.predict(X_test)

        # Log predictions and actual values for debugging
        self.log_action(f"First 5 predictions:\n{predictions[:5]}", "debug")
        self.log_action(f"First 5 actual values (y_test):\n{y_test[:5].values}", "debug")

        # Compare predictions to actual values and calculate error metrics
        mae = random_forest_model.evaluate(X_test, y_test)
        self.log_action(f"Mean Absolute Error (MAE) on test set: {mae:.4f}", "info")

        # Get the last prediction and actual close price
        last_pred = predictions[-1]
        last_close = y_test.iloc[-1]

        # Calculate the difference between predicted and actual close price
        price_diff = abs(last_pred - last_close)
        msg = f"Predicted Price: {last_pred:.2f}, Actual Close: {last_close:.2f}"

        # Define a threshold for HOLD signal (e.g., if difference is < 0.5%)
        hold_threshold = 0.005 * last_close

        # Trading signal logic with SMA/EMA confirmation
        if price_diff < hold_threshold:
            decision = "HOLD"
            self.log_action(f"{msg} -> {decision} signal (price difference: {price_diff:.2f} < threshold: {hold_threshold:.2f})", "warning")
            send_email(f"{decision} Signal for {asset} using {strategy_name}",
                       f"{msg} -> {decision} signal", self.config['notification_email'])
        elif last_pred > last_close:
            decision = "BUY"
            self.log_action(f"{msg} -> {decision} signal (predicted price higher and SMA/EMA support BUY)", "info")
            send_email(f"{decision} Signal for {asset} using {strategy_name}",
                       f"{msg} -> {decision} signal", self.config['notification_email'])
        else:
            decision = "SELL"
            self.log_action(f"{msg} -> {decision} signal (predicted price lower and SMA/EMA support SELL)", "info")
            send_email(f"{decision} Signal for {asset} using {strategy_name}",
                       f"{msg} -> {decision} signal", self.config['notification_email'])
