from strategies.strategy_base import StrategyBase

from models.lstm_model import create_lstm_model  # Use existing model class
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.email_notifications import send_email
from indicators.rsi import RSI


class DayTradingStrategy(StrategyBase):
    def execute(self):
        # Log asset and strategy information
        asset = self.config["ticker"]
        strategy_name = self.config["strategy"]

        self.log_action(f"Executing {strategy_name} strategy for {asset} using LSTM model", "info")

        # Limit data to recent rows (e.g., last 30 days)
        recent_data = self.data.tail(720)  # Assuming 24 data points per day (1-hour intervals), take last 30 days
        self.log_action(f"Using {recent_data.shape[0]} most recent rows for training and testing.", "info")

        # Log the most recent close price fetched from the data
        self.log_action(f"Most recent close price from data: {recent_data['Close'].iloc[-1]}", "info")

        if "rsi" in self.config["indicators"]:
            self.log_action("Calculating RSI indicator...", "info")
            rsi_indicator = RSI(recent_data)
            recent_data['RSI'] = rsi_indicator.calculate()

        X = np.array(recent_data[['Close', 'RSI']])
        y = np.array(recent_data['Close'])
        X = np.nan_to_num(X)

        # Perform a time-based split: use the first 80% for training and the last 20% for testing
        split_index = int(0.8 * len(X))  # 80% training, 20% testing
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Apply scaling after splitting to avoid data leakage
        self.log_action("Scaling the data using MinMaxScaler...", "info")
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        # Fit scaler on the training data, and apply to both train and test sets
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

        # Debugging: Log scaled y_test values for verification
        self.log_action(f"Scaled y_test values (first 5): {y_test_scaled[:5]}", "info")

        # Define model-building function using create_lstm_model only once
        input_shape = (X_train_scaled.shape[1], 1)

        def model_builder(units=50, learning_rate=0.001, optimizer_type='adam'):
            return create_lstm_model(input_shape=input_shape, units=units, learning_rate=learning_rate, optimizer_type=optimizer_type)

        # Wrap the create_lstm_model method with KerasRegressor for grid search
        model = KerasRegressor(model=model_builder)

        # Randomized search parameter grid
        param_grid = {
            'model__units': [50, 100],
            'model__learning_rate': [0.001, 0.0001],
            'model__optimizer_type': ['adam', 'rmsprop'],
            'batch_size': [32, 64],
            'epochs': [50, 100]
        }

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Use RandomizedSearchCV instead of GridSearchCV to limit the number of iterations
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5, cv=3)

        self.log_action("Running RandomizedSearchCV for hyperparameter tuning...", "info")
        random_search_result = random_search.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled), callbacks=[early_stopping])

        self.log_action(f"Best parameters from RandomizedSearchCV: {random_search_result.best_params_}", "info")

        # Train final model with best parameters
        best_model = random_search_result.best_estimator_.model_

        best_model.fit(X_train_scaled, y_train_scaled, epochs=random_search_result.best_params_['epochs'], batch_size=random_search_result.best_params_['batch_size'], validation_data=(X_test_scaled, y_test_scaled), callbacks=[early_stopping])

        self.log_action("Making predictions with the best model...", "info")
        y_pred_scaled = best_model.predict(X_test_scaled)

        # Debugging: Log scaled predicted values before inverse transforming
        self.log_action(f"Scaled y_pred values (first 5): {y_pred_scaled[:5]}", "info")

        # Inverse-transform both the actual and predicted y values to original scale
        y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)  # Inverse transform actual values
        y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))  # Inverse transform predicted values

        # Debugging: Log inverse-transformed values for verification
        self.log_action(f"Inverse-transformed y_test values (first 5): {y_test_rescaled[:5]}", "info")
        self.log_action(f"Inverse-transformed y_pred values (first 5): {y_pred_rescaled[:5]}", "info")

        # Decision-making logic (buy/sell/hold signals)
        last_rsi_value = recent_data['RSI'].iloc[-1]
        last_pred = y_pred_rescaled[-1][0]
        last_close = y_test_rescaled[-1][0]
        msg = f"RSI: {last_rsi_value}, Predicted Close: {last_pred}, Actual Close: {last_close}"

        if last_rsi_value < 30 and last_pred > last_close:
            decision = "BUY"
        elif last_rsi_value > 70 and last_pred < last_close:
            decision = "SELL"
        else:
            decision = "HOLD"

        self.log_action(f"{msg} -> {decision} signal", "info")
        send_email(f"{decision} Signal for {asset} using {strategy_name}", f"{msg} -> {decision} signal", self.config['notification_email'])

