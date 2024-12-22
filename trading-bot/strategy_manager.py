def select_strategy(config, data):
    """
    Select the appropriate strategy class based on the strategy type.
    """
    if config["strategy"] == "day_trading":
        from strategies.day_trading_strategy import DayTradingStrategy
        return DayTradingStrategy(config, data)
    elif config["strategy"] == "short_term":
        from strategies.short_term_strategy import ShortTermStrategy
        return ShortTermStrategy(config, data)
    elif config["strategy"] == "mid_term":
        from strategies.mid_term_strategy import MidTermStrategy
        return MidTermStrategy(config, data)
    elif config["strategy"] == "long_term":
        from strategies.long_term_strategy import LongTermStrategy
        return LongTermStrategy(config, data)
