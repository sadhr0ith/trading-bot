"""Multi-asset portfolio backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from binance.client import Client

from trading_bot.backtest.costs import fee_from_notional, fill_price, split_notional_for_fee
from trading_bot.backtest.interfaces import Signal, StrategyState, Trade
from trading_bot.backtest.metrics import compute_metrics
from trading_bot.models.env_settings import load_binance_settings
from trading_bot.utils.logger import setup_logger

logger = setup_logger("PortfolioBacktest")

_STABLE_BASES = {"USDT", "BUSD", "USDC", "TUSD", "FDUSD", "USDP", "DAI"}


@dataclass
class PortfolioConstraints:
    max_positions: int = 5
    max_exposure_per_asset: float = 0.2
    vol_targeting: bool = True
    vol_window: int = 20


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict[str, float | None]


def load_top_volume_universe(
    limit: int = 10,
    quote_asset: str = "USDT",
    cache_path: str | Path = "cache/universe_top_volume.json",
    cache_ttl_seconds: int = 3600,
) -> list[str]:
    path = Path(cache_path)
    if path.exists():
        cached_time = pd.to_datetime(path.stat().st_mtime, unit="s")
        age = max(0.0, (pd.Timestamp.utcnow() - cached_time).total_seconds())
        if age < cache_ttl_seconds:
            try:
                with path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                return list(payload.get("symbols", []))
            except (OSError, ValueError, TypeError):
                pass

    settings = load_binance_settings(logger)
    api_key = settings.api_key if settings else None
    api_secret = settings.api_secret if settings else None
    try:
        client = Client(api_key=api_key, api_secret=api_secret)
        tickers = client.get_ticker()
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning(f"Failed to fetch Binance universe: {exc}")
        if path.exists():
            try:
                with path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                return list(payload.get("symbols", []))
            except (OSError, ValueError, TypeError):
                return []
        return []

    filtered = []
    for item in tickers:
        symbol = item.get("symbol", "")
        if not symbol.endswith(quote_asset):
            continue
        base = symbol[: -len(quote_asset)]
        if base in _STABLE_BASES:
            continue
        try:
            vol = float(item.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            vol = 0.0
        filtered.append((symbol, vol))

    filtered.sort(key=lambda x: x[1], reverse=True)
    universe = [symbol for symbol, _ in filtered[:limit]]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"symbols": universe, "as_of": pd.Timestamp.utcnow().isoformat()}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except (OSError, ValueError):
        logger.warning("Failed to write universe cache.")

    return universe


def _prepare_asset_frame(df: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    out = df.copy().sort_index()
    returns = out["Close"].pct_change()
    out["Volatility"] = returns.rolling(vol_window).std()
    return out


def run_portfolio_backtest(
    data_by_asset: dict[str, pd.DataFrame],
    strategy_factory: Callable[[str, pd.DataFrame], object],
    initial_cash: float = 100_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0002,
    constraints: PortfolioConstraints | None = None,
) -> PortfolioResult:
    if not data_by_asset:
        raise ValueError("No asset data provided for portfolio backtest.")

    constraints = constraints or PortfolioConstraints()

    prepared_frames: dict[str, pd.DataFrame] = {}
    strategies: dict[str, object] = {}
    states: dict[str, StrategyState] = {}

    for ticker, df in data_by_asset.items():
        if df is None or df.empty:
            continue
        prepared = _prepare_asset_frame(df, constraints.vol_window)
        strategy = strategy_factory(ticker, prepared)
        prepare = getattr(strategy, "prepare_data", None)
        if callable(prepare):
            prepared = prepare(prepared)
        prepared_frames[ticker] = prepared
        strategies[ticker] = strategy
        states[ticker] = StrategyState(cash=0.0, position=0.0, equity=0.0)

    if not prepared_frames:
        raise ValueError("All asset frames are empty.")

    all_index = sorted(set().union(*(df.index for df in prepared_frames.values())))

    cash = float(initial_cash)
    trades: list[Trade] = []
    equity_values: list[float] = []
    index_values: list[pd.Timestamp] = []

    for ts in all_index:
        signals: dict[str, Signal] = {}
        prices: dict[str, float] = {}
        vols: dict[str, float] = {}

        for ticker, df in prepared_frames.items():
            if ts not in df.index:
                continue
            window_df = df.loc[:ts]
            row = window_df.iloc[-1]
            price = float(row["Close"])
            prices[ticker] = price
            vols[ticker] = float(row.get("Volatility", np.nan))

            state = states[ticker]
            state.current_time = ts
            signals[ticker] = strategies[ticker].on_bar(state, window_df)

        # Execute sells first
        for ticker, signal in signals.items():
            state = states[ticker]
            if signal != Signal.SELL or state.position <= 0:
                continue
            price = prices[ticker]
            fill = fill_price(price, "SELL", slippage_rate)
            notional = state.position * fill
            fee = fee_from_notional(notional, fee_rate)
            cash += notional - fee
            entry_value = (state.entry_price or 0.0) * state.position
            pnl = notional - entry_value - state.entry_fee - fee
            return_pct = pnl / entry_value if entry_value else 0.0
            trades.append(
                Trade(
                    entry_time=state.entry_time or ts,
                    exit_time=ts,
                    entry_price=state.entry_price or 0.0,
                    exit_price=fill,
                    size=state.position,
                    entry_fee=state.entry_fee,
                    exit_fee=fee,
                    entry_value=entry_value,
                    exit_value=notional,
                    pnl=pnl,
                    return_pct=return_pct,
                )
            )
            state.position = 0.0
            state.entry_price = None
            state.entry_time = None
            state.entry_fee = 0.0

        # Execute buys
        open_positions = sum(1 for s in states.values() if s.position > 0)
        slots = max(0, constraints.max_positions - open_positions)
        if slots > 0:
            candidates = [
                ticker for ticker, signal in signals.items() if signal == Signal.BUY and states[ticker].position <= 0
            ]
            if candidates:
                # Rank by absolute recent return
                scores = []
                for ticker in candidates:
                    df = prepared_frames[ticker]
                    window = df.loc[:ts]["Close"].pct_change().iloc[-1]
                    score = abs(window) if pd.notna(window) else 0.0
                    scores.append((ticker, score))
                scores.sort(key=lambda x: x[1], reverse=True)
                selected = [ticker for ticker, _ in scores[:slots]]

                if constraints.vol_targeting:
                    inv_vol = {
                        ticker: 1.0 / max(vols.get(ticker, np.nan), 1e-8)
                        for ticker in selected
                    }
                    total_inv = sum(inv_vol.values())
                    weights = {ticker: inv_vol[ticker] / total_inv if total_inv else 1.0 / len(selected) for ticker in selected}
                else:
                    weights = {ticker: 1.0 / len(selected) for ticker in selected}

                total_equity = cash + sum(
                    states[ticker].position * prices.get(ticker, 0.0) for ticker in states
                )
                for ticker in selected:
                    alloc = min(cash * weights[ticker], total_equity * constraints.max_exposure_per_asset)
                    if alloc <= 0:
                        continue
                    notional = split_notional_for_fee(alloc, fee_rate)
                    fill = fill_price(prices[ticker], "BUY", slippage_rate)
                    if fill <= 0:
                        continue
                    size = notional / fill
                    fee = fee_from_notional(notional, fee_rate)
                    total_cost = notional + fee
                    if total_cost > cash:
                        continue
                    cash -= total_cost
                    state = states[ticker]
                    state.position = size
                    state.entry_price = fill
                    state.entry_time = ts
                    state.entry_fee = fee

        equity = cash + sum(states[ticker].position * prices.get(ticker, 0.0) for ticker in states)
        equity_values.append(equity)
        index_values.append(ts)

    equity_curve = pd.Series(equity_values, index=pd.to_datetime(index_values), name="equity")
    metrics = compute_metrics(equity_curve, trades)
    return PortfolioResult(equity_curve=equity_curve, trades=trades, metrics=metrics)
