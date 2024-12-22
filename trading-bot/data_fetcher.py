import yfinance as yf
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone
import logging
import re

logger = logging.getLogger("TradingBot")

def parse_period_to_timedelta(period):
    match = re.match(r"(\d+)([a-zA-Z]+)", period)
    if not match:
        raise ValueError(f"Invalid period format: {period}")

    value, unit = match.groups()
    value = int(value)
    
    if unit == 'mo':
        return timedelta(days=value * 30)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    else:
        raise ValueError(f"Unknown period unit: {unit}")

def fetch_yahoo_data(ticker, period, interval):
    try:
        logger.info(f"Fetching Yahoo Finance data for {ticker} with period '{period}' and interval '{interval}'...")
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            logger.warning("No data returned from Yahoo Finance.")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance data: {str(e)}")
        return pd.DataFrame()

def fetch_binance_data(ticker, interval, period):
    try:
        client = Client(api_key='your_api_key', api_secret='your_api_secret')
        all_klines = fetch_binance_klines(client, ticker, interval, period)

        if not all_klines:
            logger.warning("No data returned from Binance.")
            return pd.DataFrame()

        data = pd.DataFrame(all_klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data['Close time'] = pd.to_datetime(data['Close time'], unit='ms')
        data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric)

        print(f"Downloaded {len(data)} rows of crypto data for {ticker} from Binance:")
        print(data.head())
        print(data.tail())
        
        return data
    
    except Exception as e:
        logger.error(f"Error fetching Binance data: {str(e)}")
        return pd.DataFrame()

def fetch_binance_klines(client, ticker, interval, period):
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
        klines = client.get_klines(symbol=ticker, interval=interval, startTime=start_time_ms, limit=batch_size)

        if not klines:
            break

        all_klines.extend(klines)
        last_kline = klines[-1][0]

        next_start_time = datetime.fromtimestamp(last_kline / 1000, tz=timezone.utc) + interval_to_timedelta[interval]

        if next_start_time >= end_time or len(klines) < batch_size:
            break
        else:
            start_time = next_start_time

    return all_klines

def fetch_data_online(source='yahoo', ticker='BTCUSDT', period='1y', interval='1d'):
    if source == 'yahoo':
        return fetch_yahoo_data(ticker, period, interval)
    elif source == 'binance':
        return fetch_binance_data(ticker, interval, period)
    else:
        logger.error(f"Invalid data source: {source}")
        return pd.DataFrame()
