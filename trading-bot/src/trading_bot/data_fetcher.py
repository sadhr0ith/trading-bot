import hashlib
import logging
import os
import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
try:
    import pybreaker
except ModuleNotFoundError:  # pragma: no cover - fallback for missing optional dependency
    class _BreakerState:
        def __init__(self, name: str):
            self.name = name

    class _FallbackCircuitBreakerError(Exception):
        pass

    class _FallbackCircuitBreakerListener:
        def state_change(self, cb, old_state, new_state):
            pass

        def failure(self, cb, exc):
            pass

    class _FallbackCircuitBreaker:
        def __init__(self, fail_max=5, reset_timeout=60, name="circuit", listeners=None):
            self.fail_max = fail_max
            self.reset_timeout = reset_timeout
            self.name = name
            self._listeners = listeners or []
            self.current_state = "closed"
            self._failure_count = 0
            self._opened_since = None

        def __call__(self, func):
            def wrapper(*args, **kwargs):
                if self.current_state == "open":
                    if self._opened_since is not None and (time.time() - self._opened_since) >= self.reset_timeout:
                        self.half_open()
                    else:
                        raise _FallbackCircuitBreakerError(f"Circuit '{self.name}' is open")
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    self._record_failure(exc)
                    raise
                else:
                    if self.current_state == "half_open":
                        self.close()
                    else:
                        self._failure_count = 0
                    return result

            return wrapper

        def _record_failure(self, exc):
            for listener in self._listeners:
                if hasattr(listener, "failure"):
                    listener.failure(self, exc)
            if self.current_state == "half_open":
                self.open()
                return
            self._failure_count += 1
            if self._failure_count >= self.fail_max:
                self.open()

        def _set_state(self, new_state: str):
            old_state = self.current_state
            if old_state == new_state:
                return
            self.current_state = new_state
            self._opened_since = time.time() if new_state == "open" else None
            old_obj = _BreakerState(old_state)
            new_obj = _BreakerState(new_state)
            for listener in self._listeners:
                if hasattr(listener, "state_change"):
                    listener.state_change(self, old_obj, new_obj)

        def open(self):
            self._set_state("open")

        def close(self):
            self._failure_count = 0
            self._set_state("closed")

        def half_open(self):
            self._set_state("half_open")

    class _FallbackPybreakerModule:
        CircuitBreaker = _FallbackCircuitBreaker
        CircuitBreakerListener = _FallbackCircuitBreakerListener
        CircuitBreakerError = _FallbackCircuitBreakerError

    pybreaker = _FallbackPybreakerModule()
import requests.exceptions
import yfinance as yf
from binance.client import Client
from binance.exceptions import BinanceAPIException

from trading_bot.models.env_settings import RuntimeSettings, load_binance_settings, load_cache_settings
from trading_bot.utils.logger import get_logger
from trading_bot.utils.time_utils import interval_to_timedelta, parse_period_to_timedelta

logger = get_logger(__name__)


# Circuit breakers for API resilience
# Opens after 5 consecutive failures, resets after 60 seconds
class _CircuitBreakerListener(pybreaker.CircuitBreakerListener):
    """Log circuit breaker state changes."""

    def state_change(self, cb, old_state, new_state):
        logger.warning(f"Circuit breaker '{cb.name}' state: {old_state.name} -> {new_state.name}")

    def failure(self, cb, exc):
        logger.debug(f"Circuit breaker '{cb.name}' recorded failure: {exc}")


_breaker_listener = _CircuitBreakerListener()

binance_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="binance_api",
    listeners=[_breaker_listener],
)

yahoo_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    name="yahoo_api",
    listeners=[_breaker_listener],
)


@yahoo_circuit_breaker
def _yahoo_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Wrapper around yf.download protected by circuit breaker.

    Raises:
        pybreaker.CircuitBreakerError: When circuit is open (too many recent failures)
        Exception: Any error from yfinance API
    """
    return yf.download(ticker, period=period, interval=interval, progress=False, threads=False, timeout=20)


@binance_circuit_breaker
def _binance_get_klines(client: Client, ticker: str, interval: str, start_time_ms: int, batch_size: int) -> list:
    """Wrapper around client.get_klines protected by circuit breaker.

    Raises:
        pybreaker.CircuitBreakerError: When circuit is open (too many recent failures)
        Exception: Any error from Binance API
    """
    return client.get_klines(symbol=ticker, interval=interval, startTime=start_time_ms, limit=batch_size)


def _effective_cache_ttl(interval: str | None, cache_ttl: int) -> int:
    """Cap cache TTL to a small multiple of interval to avoid serving stale candles."""
    if cache_ttl <= 0:
        return 0
    delta = interval_to_timedelta(interval)
    if not delta:
        return cache_ttl
    interval_seconds = int(delta.total_seconds())
    return min(cache_ttl, max(interval_seconds, 60) * 2)


def _resolve_cache_ttl(interval: str | None, cache_ttl_seconds: int | None, settings: RuntimeSettings | None) -> int:
    base_ttl = cache_ttl_seconds
    if base_ttl is None and settings:
        base_ttl = settings.cache_ttl_seconds
    base_ttl = base_ttl or 0
    return _effective_cache_ttl(interval, base_ttl)


def _interval_to_pandas_rule(interval: str) -> str | None:
    mapping = {
        "1m": "1T",
        "5m": "5T",
        "15m": "15T",
        "30m": "30T",
        "90m": "90T",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
        "1w": "1W",
        "1wk": "1W",
        "1mo": "30D",  # approximate
    }
    return mapping.get(interval)


def _resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = _interval_to_pandas_rule(interval)
    if not rule:
        return df
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df.resample(rule).agg(agg).dropna(how="any")


def _normalize_ohlcv(
    df: pd.DataFrame, expected_interval: str | None = None, resample_on_mismatch: bool = False
) -> pd.DataFrame:
    """Normalize OHLCV frame: timezone, sorting, dedup, numeric types, interval sanity check."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Ensure DatetimeIndex, tz-aware UTC
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except (ValueError, TypeError):
            logger.error("Fetched data index is not datetime; dropping frame.")
            return pd.DataFrame()

    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    # Deduplicate and sort
    if out.index.duplicated().any():
        dup_count = int(out.index.duplicated().sum())
        logger.warning(f"Found {dup_count} duplicate index entries; keeping first occurrence.")
        out = out[~out.index.duplicated(keep="first")]
    if not out.index.is_monotonic_increasing:
        logger.warning("Index not sorted; sorting chronologically.")
        out = out.sort_index()

    # Coerce OHLCV to numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop rows with NaN in required OHLCV columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    available_required = [c for c in required_cols if c in out.columns]
    before_drop = len(out)
    out = out.dropna(subset=available_required)
    dropped = before_drop - len(out)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with NaN in required OHLCV columns after normalization.")

    # Interval sanity check
    if expected_interval:
        expected_delta = interval_to_timedelta(expected_interval)
        if expected_delta:
            diffs = out.index.to_series().diff().dropna()
            if not diffs.empty:
                median_diff = diffs.median()
                gap_count = int((diffs > expected_delta * 1.5).sum())
                if abs(median_diff - expected_delta) > expected_delta * 0.25 or gap_count > 0:
                    warn_msg = (
                        f"Detected interval mismatch: expected {expected_interval} "
                        f"({expected_delta}), median diff {median_diff}, gaps>{expected_delta*1.5}: {gap_count}."
                    )
                    logger.warning(warn_msg)
                    # Also emit on root logger so caplog (root) captures in tests.
                    logging.getLogger().warning(warn_msg)
                    if resample_on_mismatch:
                        out = _resample_ohlcv(out, expected_interval)
                        logger.info(f"Resampled OHLCV to {expected_interval} after mismatch detection.")

    return out


def fetch_yahoo_data(ticker, period, interval):
    try:
        logger.info(f"Fetching Yahoo Finance data for {ticker} with period '{period}' and interval '{interval}'...")
        data = _yahoo_download(ticker, period, interval)
        if data.empty:
            logger.warning("No data returned from Yahoo Finance.")
            return pd.DataFrame()

        return data
    except pybreaker.CircuitBreakerError:
        logger.error(f"Yahoo Finance circuit breaker open; skipping API call for {ticker}")
        return pd.DataFrame()
    except (ConnectionError, TimeoutError, requests.exceptions.RequestException, OSError) as e:
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

    # Verify index is DatetimeIndex (assert is not reliable under PYTHONOPTIMIZE)
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except (ValueError, TypeError):
            logger.error("Index is not DatetimeIndex and could not be coerced; dropping fetched frame.")
            return pd.DataFrame()

    # Normalize timezone: Binance timestamps are UTC
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
        logger.debug(f"Localized Binance data index to UTC for {ticker}")

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

    # Start slightly before the last cached candle to avoid gaps (dedup later)
    start_time = since_time - interval_to_timedelta.get(interval, timedelta(minutes=1))
    if start_time < datetime.fromtimestamp(0, tz=timezone.utc):
        start_time = datetime.fromtimestamp(0, tz=timezone.utc)
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


def fetch_binance_data(
    ticker,
    interval,
    period,
    enable_incremental=True,
    resample_on_mismatch: bool = False,
    cache_ttl_seconds: int | None = None,
):
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
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if settings:
        api_key = settings.api_key or api_key
        api_secret = settings.api_secret or api_secret

    if not (api_key and api_secret):
        logger.info("Binance API keys not set; using public endpoints for market data.")

    # Load cache for incremental fetch
    cache_settings = load_cache_settings(logger)
    cache_ttl = _resolve_cache_ttl(interval, cache_ttl_seconds, cache_settings)
    cache_path = _cache_paths("binance", ticker, period, interval)
    cached_df = None

    if enable_incremental and cache_ttl > 0:
        cached_df = _load_cache(cache_path, cache_ttl)

    try:
        client = Client(api_key=api_key, api_secret=api_secret)

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

                normalized = _normalize_ohlcv(
                    combined_df,
                    expected_interval=interval,
                    resample_on_mismatch=resample_on_mismatch,
                )

                # Save updated cache
                if cache_ttl > 0:
                    _save_cache(cache_path, normalized)

                return normalized
            else:
                age = _cache_age_seconds(cache_path)
                logger.info(
                    f"No new data available, using cached data (age={age:.0f}s, ttl={cache_ttl}s)"
                    if age is not None
                    else "No new data available, using cached data"
                )
                return _normalize_ohlcv(cached_df, expected_interval=interval, resample_on_mismatch=resample_on_mismatch)

        # Full fetch (no cache or incremental disabled)
        logger.info(f"Full fetch for {ticker} (period={period}, interval={interval})")
        all_klines = fetch_binance_klines(client, ticker, interval, period)

        if not all_klines:
            logger.warning("No data returned from Binance.")
            return pd.DataFrame()

        data = _binance_klines_to_dataframe(all_klines, ticker)

        normalized = _normalize_ohlcv(data, expected_interval=interval, resample_on_mismatch=resample_on_mismatch)

        # Save cache
        if cache_ttl > 0:
            _save_cache(cache_path, normalized)

        return normalized

    except pybreaker.CircuitBreakerError:
        logger.error(f"Binance circuit breaker open; API unavailable for {ticker}")
        # Fallback to cache if available
        if cached_df is not None:
            age = _cache_age_seconds(cache_path)
            logger.warning(
                f"Using stale cache due to circuit breaker (age={age:.0f}s)" if age is not None else "Using stale cache due to circuit breaker"
            )
            return _normalize_ohlcv(cached_df, expected_interval=interval, resample_on_mismatch=resample_on_mismatch)
        return pd.DataFrame()
    except (ConnectionError, TimeoutError, requests.exceptions.RequestException, BinanceAPIException, OSError) as e:
        logger.error(f"Error fetching Binance data: {str(e)}")
        # Fallback to cache if available
        if cached_df is not None:
            age = _cache_age_seconds(cache_path)
            logger.warning(
                f"Using cache due to fetch error (age={age:.0f}s, ttl={cache_ttl}s)" if age is not None else "Using cache due to fetch error"
            )
            return _normalize_ohlcv(cached_df, expected_interval=interval, resample_on_mismatch=resample_on_mismatch)
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
    """Fetch klines with circuit breaker, retry/backoff and rate-limit aware logging.

    Circuit breaker opens after 5 consecutive failures across all callers,
    preventing cascade failures. Retries happen within closed/half-open state.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return _binance_get_klines(client, ticker, interval, start_time_ms, batch_size)
        except pybreaker.CircuitBreakerError:
            # Circuit is open - fail fast without retrying
            logger.error(f"Binance circuit breaker open; skipping API call for {ticker}")
            raise
        except (ConnectionError, TimeoutError, requests.exceptions.RequestException, BinanceAPIException, OSError) as exc:
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
    except ValueError:
        # Fallback to raw period string if parsing fails
        period_seconds = period
        logger.warning(f"Failed to normalize period '{period}' for cache key; using raw value")

    # Cache version allows invalidation when format changes
    # v2: Added gzip compression for 5-10x smaller cache files
    cache_version = "v2"

    # Build cache key with all parameters that affect the fetched data
    raw_key = f"{cache_version}:{source}:{ticker}:{period_seconds}:{interval}"

    # Use SHA256 for better collision resistance and security
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    return cache_dir / f"{key_hash}.pkl.gz"


def _cache_age_seconds(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None


def _load_cache(path: Path, ttl_seconds: int):
    """Load cached DataFrame with gzip compression support.

    Args:
        path: Path to cache file (.pkl.gz)
        ttl_seconds: Cache TTL in seconds (0 = disabled)

    Returns:
        DataFrame if cache is valid and not expired, None otherwise
    """
    if ttl_seconds <= 0 or not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        df = pd.read_pickle(path, compression="gzip")
        logger.info(f"Loaded cache {path.name} (age={age:.0f}s, ttl={ttl_seconds}s, compressed).")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
            logger.warning(f"Cache {path.name} had naive DatetimeIndex; localized to UTC.")
        return df
    except (OSError, pickle.UnpicklingError, ValueError):
        return None


def _save_cache(path: Path, df: pd.DataFrame):
    """Save DataFrame to cache with gzip compression.

    Compression reduces cache file size by 5-10x, improving I/O performance.

    Args:
        path: Path to cache file (.pkl.gz)
        df: DataFrame to cache
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path, compression="gzip")
        # Log file size for monitoring compression effectiveness
        size_kb = path.stat().st_size / 1024
        logger.info(f"Saved cache to {path} ({len(df)} rows, {size_kb:.1f} KB compressed).")
    except OSError:
        logger.warning(f"Failed to write cache {path}")


def fetch_data_online(
    source="yahoo",
    ticker="BTCUSDT",
    period="1y",
    interval="1d",
    cache_ttl_seconds: int | None = None,
    resample_on_mismatch: bool = True,
):
    cache_settings = load_cache_settings(logger)
    cache_ttl = _resolve_cache_ttl(interval, cache_ttl_seconds, cache_settings)
    cache_path = _cache_paths(source, ticker, period, interval)

    cached = _load_cache(cache_path, cache_ttl)
    if cached is not None:
        age = _cache_age_seconds(cache_path)
        ttl_note = f"(age={age:.0f}s, ttl={cache_ttl}s)" if age is not None else f"(ttl={cache_ttl}s)"
        logger.info(f"Serving {source}:{ticker} {period}/{interval} from cache {ttl_note}.")
        return _normalize_ohlcv(cached, expected_interval=interval, resample_on_mismatch=resample_on_mismatch)

    if source == "yahoo":
        df = fetch_yahoo_data(ticker, period, interval)
    elif source == "binance":
        df = fetch_binance_data(
            ticker,
            interval,
            period,
            resample_on_mismatch=resample_on_mismatch,
            cache_ttl_seconds=cache_ttl,
        )
    else:
        logger.error(f"Invalid data source: {source}")
        return pd.DataFrame()

    normalized = _normalize_ohlcv(df, expected_interval=interval, resample_on_mismatch=resample_on_mismatch)

    if cache_ttl > 0 and not normalized.empty and source != "binance":
        _save_cache(cache_path, normalized)

    return normalized
