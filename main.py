import os
import json
import time
import math
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# -------------------------
# Environment / Settings
# -------------------------
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
APCA_DATA_BASE_URL = os.environ.get("APCA_DATA_BASE_URL", "https://data.alpaca.markets")
APCA_DATA_FEED = (os.environ.get("APCA_DATA_FEED", "iex").lower())  # "iex" or "sip"

GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Trading Log")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON", "")

REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", 30))
# Bigger batches are fine now because we fetch bars for the whole batch at once
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 200))
SLEEP_BETWEEN_BATCHES = int(os.environ.get("SLEEP_BETWEEN_BATCHES", 1))

# Require at least this many 15m bars to keep a symbol
MIN_15M_BARS = int(os.environ.get("MIN_15M_BARS", 20))

TZ = pytz.UTC

SCREENER_TAB = "Alpaca-Screener"
CHARTDATA_TAB = "Alpaca-Screener-chartData"

# -------------------------
# Utilities / Indicators
# -------------------------
def utcnow():
    return datetime.now(timezone.utc)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

# -------------------------
# Clients
# -------------------------
trading_client = TradingClient(
    APCA_API_KEY_ID, APCA_API_SECRET_KEY,
    paper=APCA_API_BASE_URL.endswith("paper-api.alpaca.markets")
)
stock_data_client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

# -------------------------
# Google Sheets helpers
# -------------------------
def get_gspread_client():
    if not GOOGLE_CREDS_JSON:
        raise RuntimeError("GOOGLE_CREDS_JSON env var is empty. Provide service account JSON.")
    info = json.loads(GOOGLE_CREDS_JSON)
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)

def open_or_create_sheet(gc) -> gspread.Spreadsheet:
    try:
        return gc.open(GOOGLE_SHEET_NAME)
    except gspread.SpreadsheetNotFound:
        return gc.create(GOOGLE_SHEET_NAME)

def ensure_worksheet(sh, title: str, rows: int = 100, cols: int = 50) -> gspread.Worksheet:
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))

# -------------------------
# Asset list (REST; filter to primary exchanges)
# -------------------------
ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "BATS", "NYSEARCA"}

def list_alpaca_tradable_equities() -> List[Dict]:
    url = APCA_API_BASE_URL.rstrip('/') + "/v2/assets"
    headers = {
        "APCA-API-KEY-ID": APCA_API_KEY_ID or "",
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY or "",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        assets = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch assets via HTTP: {e}")
        return []

    out: List[Dict] = []
    for a in assets:
        symbol = a.get("symbol")
        if not symbol:
            continue
        if (a.get("status") or "").lower() != "active":
            continue
        if not a.get("tradable", False):
            continue
        cls = (a.get("class") or a.get("asset_class") or "").lower()
        if cls != "us_equity":
            continue
        exch = (a.get("exchange") or "").upper()
        if exch not in ALLOWED_EXCHANGES:
            continue
        out.append({"symbol": symbol})

    # Deduplicate
    seen = set()
    unique = []
    for a in out:
        s = a["symbol"]
        if s not in seen:
            unique.append(a)
            seen.add(s)
    return unique

# -------------------------
# Batched data fetching
# -------------------------
class TransientDataError(Exception):
    pass

@retry(
    stop=stop_after_attempt(2),  # fewer retries to keep runs fast
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(TransientDataError)
)
def fetch_stock_bars_batch(symbols: List[str], tf: TimeFrame, limit: int) -> pd.DataFrame:
    """Fetch bars for many symbols in a single request."""
    feed = DataFeed.IEX if APCA_DATA_FEED == "iex" else DataFeed.SIP
    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            limit=limit,
            feed=feed,
            adjustment=None  # raw prices; fastest
        )
        df = stock_data_client.get_stock_bars(req).df
        return df
    except Exception as e:
        msg = str(e).lower()
        transient = ["timeout", "temporarily", "rate limit", "429", "too many requests", "connection reset", "502", "503", "504"]
        permanent = ["403", "forbidden", "plan", "not entitled", "unauthorized"]
        if any(m in msg for m in permanent):
            raise
        if any(m in msg for m in transient):
            raise TransientDataError(msg)
        raise TransientDataError(msg)

def slice_symbol_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex):
        try:
            sdf = df.xs(symbol, level='symbol')
        except KeyError:
            return None
    else:
        sdf = df[df.get('symbol', '') == symbol] if 'symbol' in df.columns else df
    if sdf is None or sdf.empty:
        return None
    return sdf.sort_index()

# -------------------------
# Per-symbol computation (using batched data)
# -------------------------
def compute_metrics_from_batch(symbol: str, df_15: pd.DataFrame | None, df_1d: pd.DataFrame | None) -> Tuple[Dict, pd.Series] | Tuple[None, None]:
    s15 = slice_symbol_df(df_15, symbol)
    if s15 is None or len(s15) < MIN_15M_BARS:
        return None, None

    closes_15 = s15['close']
    volumes_15 = s15['volume'] if 'volume' in s15.columns else pd.Series(index=s15.index, data=np.nan)

    rsi14_series = rsi(closes_15, 14)
    ma60 = closes_15.rolling(window=60, min_periods=1).mean()
    ma240 = closes_15.rolling(window=240, min_periods=1).mean()
    macd_line, signal_line = macd(closes_15)

    rsi14 = float(rsi14_series.iloc[-1]) if len(rsi14_series) else np.nan
    ma60_last = float(ma60.iloc[-1]) if len(ma60) else np.nan
    ma240_last = float(ma240.iloc[-1]) if len(ma240) else np.nan
    macd_last = float(macd_line.iloc[-1]) if len(macd_line) else np.nan
    macd_signal_last = float(signal_line.iloc[-1]) if len(signal_line) else np.nan

    # 96×15m bars ~ 24h rolling window
    recent_96 = volumes_15.tail(96)
    vol_24h = float(recent_96.sum(skipna=True)) if len(recent_96) else np.nan

    s1d = slice_symbol_df(df_1d, symbol)
    pl_1d = pl_7d = pl_14d = np.nan
    pct_down_ath = np.nan
    if s1d is not None and not s1d.empty:
        last_close = s1d['close'].iloc[-1]
        def pct_change_n_days(n):
            idxmax = s1d.index.max()
            cutoff = idxmax - pd.Timedelta(days=n)
            prior = s1d[s1d.index <= cutoff]
            if prior.empty:
                return np.nan
            ref = prior['close'].iloc[-1]
            return float(((last_close - ref) / ref) * 100.0) if ref else np.nan
        pl_1d  = pct_change_n_days(1)
        pl_7d  = pct_change_n_days(7)
        pl_14d = pct_change_n_days(14)
        ath = float(s1d['close'].max())
        if ath > 0:
            pct_down_ath = float(((last_close - ath) / ath) * 100.0)

    spark_series = closes_15.tail(14 * 24 * 4)
    row = {
        "symbol": symbol,
        "class": "us_equity",
        "%down_from_ATH": round(pct_down_ath, 4) if not math.isnan(pct_down_ath) else "",
        "PL%_1d": round(pl_1d, 4) if not math.isnan(pl_1d) else "",
        "PL%_7d": round(pl_7d, 4) if not math.isnan(pl_7d) else "",
        "PL%_14d": round(pl_14d, 4) if not math.isnan(pl_14d) else "",
        "volume_24h": round(vol_24h, 4) if not math.isnan(vol_24h) else "",
        "RSI14_15m": round(rsi14, 4) if not math.isnan(rsi14) else "",
        "MA60_15m": round(ma60_last, 6) if not math.isnan(ma60_last) else "",
        "MA240_15m": round(ma240_last, 6) if not math.isnan(ma240_last) else "",
        "MACD_15m": round(macd_last, 6) if not math.isnan(macd_last) else "",
        "MACDsig_15m": round(macd_signal_last, 6) if not math.isnan(macd_signal_last) else "",
    }
    return row, spark_series

# -------------------------
# Sheets writing
# -------------------------
def write_chartdata(ws_chart: gspread.Worksheet, all_spark: Dict[str, pd.Series]):
    header = ["symbol"]
    max_len = max((len(s) for s in all_spark.values() if s is not None), default=0)
    header.extend([f"p{i+1}" for i in range(max_len)])

    values = [header]
    for sym, series in all_spark.items():
        if series is None or len(series) == 0:
            row = [sym]
        else:
            row = [sym] + list(map(lambda x: float(x) if pd.notna(x) else "", series.values))
        values.append(row)

    ws_chart.clear()
    ws_chart.update('A1', values)

def write_screener(ws_main: gspread.Worksheet, rows: List[Dict]):
    columns = [
        "symbol", "class", "%down_from_ATH", "PL%_1d", "PL%_7d", "PL%_14d",
        "volume_24h", "RSI14_15m", "MA60_15m", "MA240_15m", "MACD_15m", "MACDsig_15m", "sparkline"
    ]

    values = [columns]
    for i, r in enumerate(rows, start=2):
        base = [r.get(c, "") for c in columns[:-1]]
        formula = (
            f"=IFERROR(SPARKLINE(INDEX('{CHARTDATA_TAB}'!B:ZZ, "
            f"MATCH(A{i}, '{CHARTDATA_TAB}'!$A:$A, 0), 0)), \"\")"
        )
        base.append(formula)
        values.append(base)

    ws_main.clear()
    ws_main.update('A1', values)

# -------------------------
# Orchestration
# -------------------------
def run_once():
    gc = get_gspread_client()
    sh = open_or_create_sheet(gc)
    ws_main = ensure_worksheet(sh, SCREENER_TAB, rows=2000, cols=30)
    ws_chart = ensure_worksheet(sh, CHARTDATA_TAB, rows=2000, cols=500)

    assets = list_alpaca_tradable_equities()
    symbols = [a["symbol"] for a in assets]
    print(f"Symbols after exchange filter: {len(symbols)}")

    rows: List[Dict] = []
    spark_map: Dict[str, pd.Series] = {}

    # Precompute limits
    limit_15m = 14 * 24 * 4 + 10   # ~two weeks of 15m bars + cushion
    limit_1d  = 1300               # ~5 years of daily bars

    for i in range(0, len(symbols), BATCH_SIZE):
        batch_syms = symbols[i:i+BATCH_SIZE]
        print(f"Batch {i//BATCH_SIZE + 1}: fetching bars for {len(batch_syms)} symbols…")

        # One request for 15m, one for 1D per batch
        df_15 = None
        df_1d = None
        try:
            df_15 = fetch_stock_bars_batch(batch_syms, TimeFrame(15, TimeFrameUnit.Minute), limit_15m)
        except Exception as e:
            print(f"[WARN] 15m batch fetch failed: {e}")

        try:
            df_1d = fetch_stock_bars_batch(batch_syms, TimeFrame.Day, limit_1d)
        except Exception as e:
            print(f"[WARN] 1D batch fetch failed: {e}")

        # Compute per symbol using the batch frames
        for sym in batch_syms:
            try:
                res = compute_metrics_from_batch(sym, df_15, df_1d)
                if res is None:
                    continue
                row, spark = res
                if row is not None and spark is not None and len(spark) > 0:
                    rows.append(row)
                    spark_map[sym] = spark
            except Exception as e:
                print(f"[WARN] compute failed for {sym}: {e}")

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Order and write once
    rows.sort(key=lambda r: r.get('symbol', ''))
    write_chartdata(ws_chart, spark_map)
    write_screener(ws_main, rows)

if __name__ == "__main__":
    print("Starting Alpaca ➜ Google Sheets screener loop (batched, IEX feed)…")
    refresh_seconds = max(60, REFRESH_MINUTES * 60)
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[ERROR] run_once failed: {e}")
        print(f"Done. Sleeping for {REFRESH_MINUTES} minutes…")
        time.sleep(refresh_seconds)
