import time
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from binance.client import Client

from models.env_settings import load_binance_settings, load_cache_settings
from utils.logger import setup_logger
from utils.time_utils import parse_period_to_timedelta

logger = setup_logger("TradingBot")


def fetch_yahoo_data(ticker, period, interval):
    try:
        logger.info(f"Fetching Yahoo Finance data for {ticker} with period '{period}' and interval '{interval}'...")
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            logger.warning("No data returned from Yahoo Finance.")
            return pd.DataFrame()
        return data
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error fetching Yahoo Finance data: {str(e)}")
        return pd.DataFrame()


def _binance_klines_to_dataframe(klines, ticker):
    """Convert Binance klines to DataFrame with proper formatting and validation.

    Args:
        klines: List of kline data from Binance API
        ticker: Asset symbol for logging

    Returns:
        DataFrame with OHLCV data indexed by timestamp
    """
    if not klines:
        return pd.DataFrame()

    data = pd.DataFrame(
        klines,
        columns=[
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ],
    )

    data["Open time"] = pd.to_datetime(data["Open time"], unit="ms")
    data["Close time"] = pd.to_datetime(data["Close time"], unit="ms")
    data[["Open", "High", "Low", "Close", "Volume"]] = data[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric
    )

    # Set Open time as index for time-based operations
    data.set_index("Open time", inplace=True)
    data.index.name = "timestamp"

    # Verify index is DatetimeIndex
    assert isinstance(data.index, pd.DatetimeIndex), "Index must be DatetimeIndex for time-based operations"

    # Basic data-quality checks
    if len(data) < 50:
        logger.warning(f"Binance returned only {len(data)} rows; training/inference may be unreliable.")
    if not data.index.is_monotonic_increasing:
        logger.warning("Timestamp index is not monotonic; sorting index.")
        data = data.sort_index()
    diffs = data.index.to_series().diff().dropna()
    if not diffs.empty:
        median_diff = diffs.median()
        gap_count = (diffs > median_diff * 1.5).sum()
        if gap_count > 0:
            logger.warning(
                f"Detected {gap_count} timestamp gap(s) in Binance data; downstream features may be affected."
            )

    # Log simple data quality metrics
    nan_counts = data[["Open", "High", "Low", "Close", "Volume"]].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN counts in Binance data: {nan_counts.to_dict()}")

    logger.info(f"Converted {len(data)} rows of Binance klines for {ticker}.")
    return data


def fetch_binance_klines_since(client, ticker, interval, since_time, max_retries: int = 3, retry_backoff: float = 1.5):
    """Fetch Binance klines incrementally since a specific timestamp.

    Args:
        client: Binance client instance
        ticker: Asset symbol (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h", "1d")
        since_time: Datetime to fetch from (timezone-aware)
        max_retries: Maximum retry attempts
        retry_backoff: Backoff multiplier for retries

    Returns:
        List of klines data
    """
    all_klines = []
    batch_size = 1000
    interval_to_timedelta = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    # Add a small buffer to avoid missing the latest candle
    start_time = since_time + interval_to_timedelta.get(interval, timedelta(minutes=1))
    end_time = datetime.now(tz=timezone.utc)

    # If start time is in the future or equals end time, nothing to fetch
    if start_time >= end_time:
        logger.info(f"No new data to fetch for {ticker} (cache up to date)")
        return []

    while True:
        start_time_ms = int(start_time.timestamp() * 1000)
        klines = _get_klines_with_retry(client, ticker, interval, start_time_ms, batch_size, max_retries, retry_backoff)

        if not klines:
            break

        all_klines.extend(klines)
        last_kline = klines[-1][0]

        next_start_time = datetime.fromtimestamp(last_kline / 1000, tz=timezone.utc) + interval_to_timedelta[interval]

        if next_start_time >= end_time or len(klines) < batch_size:
            break
        start_time = next_start_time

    logger.info(f"Fetched {len(all_klines)} new klines for {ticker} since {since_time}")
    return all_klines


def fetch_binance_data(ticker, interval, period, enable_incremental=True):
    """Fetch Binance data with optional incremental updates.

    Args:
        ticker: Asset symbol (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h", "1d")
        period: Time period (e.g., "1y", "6M")
        enable_incremental: If True, use incremental fetch when cache available

    Returns:
        DataFrame with OHLCV data
    """
    settings = load_binance_settings(logger)
    if not settings:
        return pd.DataFrame()

    # Load cache for incremental fetch
    cache_settings = load_cache_settings(logger)
    cache_ttl = cache_settings.ttl_seconds if cache_settings else 0
    cache_path = _cache_paths("binance", ticker, period, interval)
    cached_df = None

    if enable_incremental and cache_ttl > 0:
        cached_df = _load_cache(cache_path, cache_ttl)

    try:
        client = Client(api_key=settings.api_key, api_secret=settings.api_secret)

        # Incremental fetch if cache available
        if enable_incremental and cached_df is not None and len(cached_df) > 0:
            latest_cached_time = cached_df.index.max()
            logger.info(f"Incremental fetch for {ticker}: last cached {latest_cached_time}")

            # Fetch only new data
            new_klines = fetch_binance_klines_since(client, ticker, interval, latest_cached_time)

            if new_klines:
                # Convert and merge
                new_df = _binance_klines_to_dataframe(new_klines, ticker)
                combined_df = pd.concat([cached_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
                combined_df = combined_df.sort_index()

                # Trim to requested period
                end_time = datetime.now(tz=timezone.utc)
                timedelta_period = parse_period_to_timedelta(period)
                start_time = end_time - timedelta_period
                combined_df = combined_df[combined_df.index >= start_time]

                logger.info(f"Incremental: +{len(new_df)} new rows, total {len(combined_df)} rows")

                # Save updated cache
                if cache_ttl > 0:
                    _save_cache(cache_path, combined_df)

                return combined_df
            else:
                logger.info("No new data available, using cached data")
                return cached_df

        # Full fetch (no cache or incremental disabled)
        logger.info(f"Full fetch for {ticker} (period={period}, interval={interval})")
        all_klines = fetch_binance_klines(client, ticker, interval, period)

        if not all_klines:
            logger.warning("No data returned from Binance.")
            return pd.DataFrame()

        data = _binance_klines_to_dataframe(all_klines, ticker)

        # Save cache
        if cache_ttl > 0:
            _save_cache(cache_path, data)

        return data

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error fetching Binance data: {str(e)}")
        # Fallback to cache if available
        if cached_df is not None:
            logger.warning("Using stale cache due to fetch error")
            return cached_df
        return pd.DataFrame()


def fetch_binance_klines(client, ticker, interval, period, max_retries: int = 3, retry_backoff: float = 1.5):
    all_klines = []
    batch_size = 1000
    interval_to_timedelta = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    end_time = datetime.now(tz=timezone.utc)
    timedelta_period = parse_period_to_timedelta(period)
    start_time = end_time - timedelta_period

    while True:
        start_time_ms = int(start_time.timestamp() * 1000)
        klines = _get_klines_with_retry(client, ticker, interval, start_time_ms, batch_size, max_retries, retry_backoff)

        if not klines:
            break

        all_klines.extend(klines)
        last_kline = klines[-1][0]

        next_start_time = datetime.fromtimestamp(last_kline / 1000, tz=timezone.utc) + interval_to_timedelta[interval]

        if next_start_time >= end_time or len(klines) < batch_size:
            break
        start_time = next_start_time

    return all_klines


def _get_klines_with_retry(client, ticker, interval, start_time_ms, batch_size, max_retries, retry_backoff):
    """Fetch klines with simple retry/backoff and rate-limit aware logging."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.get_klines(symbol=ticker, interval=interval, startTime=start_time_ms, limit=batch_size)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if "weight" in msg or "rate limit" in msg:
                logger.warning(f"Binance rate limit/weight error (attempt {attempt}/{max_retries}); backing off.")
            else:
                logger.warning(f"Binance get_klines failed (attempt {attempt}/{max_retries}): {exc}")
            if attempt < max_retries:
                sleep_seconds = retry_backoff**attempt
                time.sleep(sleep_seconds)
                continue
            logger.error(f"Exhausted retries fetching klines for {ticker}: {exc}")
            break
    if last_exc:
        raise last_exc


def _cache_paths(source: str, ticker: str, period: str, interval: str):
    """Generate cache path with robust key generation.

    Improvements over simple hashing:
    - Normalizes period to seconds for consistent keys (e.g., "1y" == "365d")
    - Uses SHA256 for better collision resistance than MD5
    - Includes cache version prefix for format invalidation
    - Ensures equivalent parameters always produce same cache key

    Args:
        source: Data source ("yahoo" or "binance")
        ticker: Asset symbol (e.g., "AAPL", "BTCUSDT")
        period: Time period (e.g., "1y", "6M", "90d")
        interval: Candle interval (e.g., "1d", "1h", "5m")

    Returns:
        Path to cache file with SHA256-based filename
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Normalize period to total seconds for consistent cache keys
    # This ensures "1y" and "365d" produce identical keys
    try:
        period_seconds = int(parse_period_to_timedelta(period).total_seconds())
    except Exception:  # noqa: BLE001
        # Fallback to raw period string if parsing fails
        period_seconds = period
        logger.warning(f"Failed to normalize period '{period}' for cache key; using raw value")

    # Cache version allows invalidation when format changes
    cache_version = "v1"

    # Build cache key with all parameters that affect the fetched data
    raw_key = f"{cache_version}:{source}:{ticker}:{period_seconds}:{interval}"

    # Use SHA256 for better collision resistance and security
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    return cache_dir / f"{key_hash}.pkl"


def _load_cache(path: Path, ttl_seconds: int):
    if ttl_seconds <= 0 or not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        return pd.read_pickle(path)
    except Exception:  # noqa: BLE001
        return None


def _save_cache(path: Path, df: pd.DataFrame):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
    except Exception:  # noqa: BLE001
        logger.warning(f"Failed to write cache {path}")


def fetch_data_online(source="yahoo", ticker="BTCUSDT", period="1y", interval="1d"):
    cache_settings = load_cache_settings(logger)
    cache_ttl = cache_settings.ttl_seconds if cache_settings else 0
    cache_path = _cache_paths(source, ticker, period, interval)

    cached = _load_cache(cache_path, cache_ttl)
    if cached is not None:
        logger.info(f"Serving {source}:{ticker} {period}/{interval} from cache (ttl={cache_ttl}s).")
        return cached

    if source == "yahoo":
        df = fetch_yahoo_data(ticker, period, interval)
    elif source == "binance":
        df = fetch_binance_data(ticker, interval, period)
    else:
        logger.error(f"Invalid data source: {source}")
        return pd.DataFrame()

    if cache_ttl > 0 and not df.empty:
        _save_cache(cache_path, df)

    return df
