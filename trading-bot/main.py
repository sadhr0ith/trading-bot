import argparse
import time
from datetime import datetime, timedelta
from config_handler import load_config, get_sleep_duration
from data_fetcher import fetch_data_online
from strategy_manager import select_strategy
from utils.logger import setup_logger

logger = setup_logger("TradingBot")

def run_trading_bot(strategy):
    """
    Main loop to run the bot continuously, updating data and retraining models.
    """
    config = load_config(strategy)
    
    if config is None:
        logger.error("Invalid strategy or configuration. Exiting.")
        return

    asset = config["ticker"]
    
    # Log the asset and strategy being used
    logger.info(f"Running {strategy} strategy for asset {asset} at {datetime.now()}")

    while True:
        logger.info(f"Fetching data for asset {asset} with strategy {strategy}")

        # Fetch market data based on the config
        data = fetch_data_online(source=config["data_source"],
                                 ticker=config["ticker"],
                                 period=config.get("period", "1y"),
                                 interval=config.get("interval", "1d"))

        if data is not None:
            # Select and run the strategy
            strategy_instance = select_strategy(config, data)
            strategy_instance.execute()

        # Get the sleep duration based on strategy timeframe
        sleep_duration = get_sleep_duration(strategy)

        logger.info(f"Sleeping for {timedelta(seconds=sleep_duration)} before fetching data again for {asset} using {strategy}.")
        time.sleep(sleep_duration)  # Sleep before fetching new data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot with the specified strategy.")
    parser.add_argument('--strategy', type=str, required=True, help="Strategy to run: 'day_trading', 'short_term', 'mid_term', 'long_term'")
    args = parser.parse_args()

    run_trading_bot(args.strategy)
