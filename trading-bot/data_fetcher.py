from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import yfinance as yf
from binance.client import Client

from models.env_settings import load_binance_settings, load_cache_settings
from utils.time_utils import parse_period_to_timedelta
from utils.logger import setup_logger

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


def fetch_binance_data(ticker, interval, period):
    settings = load_binance_settings(logger)
    if not settings:
        return pd.DataFrame()

    try:
        client = Client(api_key=settings.api_key, api_secret=settings.api_secret)
        all_klines = fetch_binance_klines(client, ticker, interval, period)

        if not all_klines:
            logger.warning("No data returned from Binance.")
            return pd.DataFrame()

        data = pd.DataFrame(
            all_klines,
            columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore',
            ],
        )

        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data['Close time'] = pd.to_datetime(data['Close time'], unit='ms')
        data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)

        # Set Open time as index for time-based operations
        data.set_index('Open time', inplace=True)
        data.index.name = 'timestamp'

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
                logger.warning(f"Detected {gap_count} timestamp gap(s) in Binance data; downstream features may be affected.")

        # Log simple data quality metrics
        nan_counts = data[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN counts in Binance data: {nan_counts.to_dict()}")

        logger.info(f"Downloaded {len(data)} rows of crypto data for {ticker} from Binance.")
        return data

    except Exception as e:  # noqa: BLE001
        logger.error(f"Error fetching Binance data: {str(e)}")
        return pd.DataFrame()


def fetch_binance_klines(client, ticker, interval, period, max_retries: int = 3, retry_backoff: float = 1.5):
    all_klines = []
    batch_size = 1000
    interval_to_timedelta = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1),
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
                sleep_seconds = retry_backoff ** attempt
                time.sleep(sleep_seconds)
                continue
            logger.error(f"Exhausted retries fetching klines for {ticker}: {exc}")
            break
    if last_exc:
        raise last_exc


def _cache_paths(source: str, ticker: str, period: str, interval: str):
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{source}_{ticker}_{period}_{interval}".replace("/", "_")
    return cache_dir / f"{key}.pkl"


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


def fetch_data_online(source='yahoo', ticker='BTCUSDT', period='1y', interval='1d'):
    cache_settings = load_cache_settings(logger)
    cache_ttl = cache_settings.ttl_seconds if cache_settings else 0
    cache_path = _cache_paths(source, ticker, period, interval)

    cached = _load_cache(cache_path, cache_ttl)
    if cached is not None:
        logger.info(f"Serving {source}:{ticker} {period}/{interval} from cache (ttl={cache_ttl}s).")
        return cached

    if source == 'yahoo':
        df = fetch_yahoo_data(ticker, period, interval)
    elif source == 'binance':
        df = fetch_binance_data(ticker, interval, period)
    else:
        logger.error(f"Invalid data source: {source}")
        return pd.DataFrame()

    if cache_ttl > 0 and not df.empty:
        _save_cache(cache_path, df)

    return df
