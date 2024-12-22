# strategies/mid_term_strategy.py
from .strategy_base import StrategyBase
from models.random_forest_model import RandomForestModel
from sklearn.model_selection import train_test_split

class MidTermStrategy(StrategyBase):
    def execute(self):
        self.log_action("Executing mid-term trading strategy with Random Forest", "info")

        # Prepare data for Random Forest (e.g., using technical indicators)
        X = self.data[['Close']]  # Example: using only Close price for simplicity
        y = self.data['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train and predict using Random Forest
        rf_model = RandomForestModel()
        rf_model.train(X_train, y_train)
        predictions = rf_model.predict(X_test)

        # Example: decision based on prediction
        last_pred = predictions[-1]
        last_close = self.data['Close'].iloc[-1]
        
        if last_pred > last_close:
            self.log_action(f"Predicted price {last_pred:.2f} is higher than current price {last_close:.2f}. BUY signal", "info")
        else:
            self.log_action(f"Predicted price {last_pred:.2f} is lower than current price {last_close:.2f}. SELL signal", "info")
