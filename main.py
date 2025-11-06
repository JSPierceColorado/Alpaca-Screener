import os
import json
import time
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential

# Alpaca
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# Kraken (we initialize to satisfy the connectivity requirement)
import krakenex

# -------------------------
# Environment / Settings
# -------------------------
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
APCA_DATA_BASE_URL = os.environ.get("APCA_DATA_BASE_URL", "https://data.alpaca.markets")

GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Trading Log")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON", "")

KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", 30))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 50))
SLEEP_BETWEEN_BATCHES = int(os.environ.get("SLEEP_BETWEEN_BATCHES", 5))

TZ = pytz.UTC

SCREENER_TAB = "Alpaca-Screener"
CHARTDATA_TAB = "Alpaca-Screener-chartData"

# -------------------------
# Utilities
# -------------------------

def utcnow():
    return datetime.now(timezone.utc)

def to_iso8601(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

# Simple technical indicators on a Series of closes
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
# Alpaca Clients
# -------------------------
trading_client = TradingClient(
    APCA_API_KEY_ID, APCA_API_SECRET_KEY,
    paper=APCA_API_BASE_URL.endswith("paper-api.alpaca.markets")
)
stock_data_client = StockHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient(APCA_API_KEY_ID, APCA_API_SECRET_KEY)

# -------------------------
# Kraken connectivity (for now we just prove the connection)
# -------------------------
kraken = krakenex.API()
if KRAKEN_API_KEY and KRAKEN_API_SECRET:
    kraken.key = KRAKEN_API_KEY
    kraken.secret = KRAKEN_API_SECRET

def ensure_kraken_connectivity():
    try:
        _ = kraken.query_public('Time')  # simple public endpoint
    except Exception as e:
        print(f"[WARN] Kraken connectivity check failed: {e}")

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
# Data Fetching
# -------------------------
def list_alpaca_tradable_assets() -> List[Dict]:
    """
    Robust to alpaca-py versions: call get_all_assets() with no kwargs,
    then filter by .tradable and status == 'active' if available.
    """
    # Some versions of alpaca-py don't accept keyword args here.
    assets = trading_client.get_all_assets()

    tradable_assets: List[Dict] = []
    for a in assets:
        # defensive attribute access across versions
        tradable = getattr(a, "tradable", False)
        status = getattr(a, "status", None)
        asset_class_obj = getattr(a, "asset_class", None)
        symbol = getattr(a, "symbol", None)

        if not symbol:
            continue
        if not tradable:
            continue
        if status is not None and str(status).lower() != "active":
            continue

        # asset_class may be an enum or string; normalize to string
        if hasattr(asset_class_obj, "value"):
            cls = asset_class_obj.value
        else:
            cls = str(asset_class_obj) if asset_class_obj is not None else ""

        tradable_assets.append({"symbol": symbol, "class": cls})

    # Deduplicate symbols defensively
    seen = set()
    unique = []
    for a in tradable_assets:
        if a["symbol"] not in seen:
            unique.append(a)
            seen.add(a["symbol"])
    return unique

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=30))
def fetch_bars(symbols: List[str], is_crypto: bool, start: datetime, end: datetime, tf: TimeFrame):
    if is_crypto:
        req = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start, end=end)
        return crypto_data_client.get_crypto_bars(req).df
    else:
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start, end=end)
        return stock_data_client.get_stock_bars(req).df

# -------------------------
# Computations per symbol
# -------------------------
def compute_metrics_for_symbol(symbol: str, cls: str) -> Tuple[Dict, pd.Series]:
    is_crypto = (cls.lower() == 'crypto')

    now = utcnow()
    two_weeks_ago = now - timedelta(days=14)
    five_years_ago = now - timedelta(days=365*5)

    # 15-min bars (2 weeks) for indicators + sparkline
    tf_15 = TimeFrame(15, TimeFrameUnit.Minute)
    try:
        df_15 = fetch_bars([symbol], is_crypto, two_weeks_ago, now, tf_15)
    except Exception as e:
        print(f"[WARN] 15m bars fetch failed for {symbol}: {e}")
        return None, None

    if df_15 is None or df_15.empty:
        return None, None

    # Ensure MultiIndex to single symbol slice handling
    if isinstance(df_15.index, pd.MultiIndex):
        try:
            df_15 = df_15.xs(symbol, level='symbol')
        except KeyError:
            return None, None

    df_15 = df_15.sort_index()

    closes_15 = df_15['close']
    volumes_15 = df_15['volume'] if 'volume' in df_15.columns else pd.Series(index=df_15.index, data=np.nan)

    # Indicators on 15-min closes
    rsi14_series = rsi(closes_15, 14)
    ma60 = closes_15.rolling(window=60, min_periods=1).mean()
    ma240 = closes_15.rolling(window=240, min_periods=1).mean()
    macd_line, signal_line = macd(closes_15)

    rsi14 = float(rsi14_series.iloc[-1]) if len(rsi14_series) else np.nan
    ma60_last = float(ma60.iloc[-1]) if len(ma60) else np.nan
    ma240_last = float(ma240.iloc[-1]) if len(ma240) else np.nan
    macd_last = float(macd_line.iloc[-1]) if len(macd_line) else np.nan
    macd_signal_last = float(signal_line.iloc[-1]) if len(signal_line) else np.nan

    # 24h volume from 15-min bars (last 96 intervals)
    recent_96 = volumes_15.tail(96)
    vol_24h = float(recent_96.sum(skipna=True)) if len(recent_96) else np.nan

    # Daily bars for P/L % and ATH (fetch 5y to approximate ATH)
    tf_1d = TimeFrame.Day
    try:
        df_1d = fetch_bars([symbol], is_crypto, five_years_ago, now, tf_1d)
        if isinstance(df_1d.index, pd.MultiIndex):
            df_1d = df_1d.xs(symbol, level='symbol')
        df_1d = df_1d.sort_index()
    except Exception as e:
        print(f"[WARN] 1D bars fetch failed for {symbol}: {e}")
        df_1d = pd.DataFrame()

    pl_1d = pl_7d = pl_14d = np.nan
    pct_down_ath = np.nan

    if not df_1d.empty:
        last_close = df_1d['close'].iloc[-1]
        def pct_change_n_days(n):
            # pick the closest trading day n days ago
            cutoff = df_1d.index.max() - pd.Timedelta(days=n)
            prior = df_1d[df_1d.index <= cutoff]
            if prior.empty:
                return np.nan
            ref = prior['close'].iloc[-1]
            return float(((last_close - ref) / ref) * 100.0) if ref else np.nan

        pl_1d = pct_change_n_days(1)
        pl_7d = pct_change_n_days(7)
        pl_14d = pct_change_n_days(14)

        # ATH based on last ~5y of daily bars (approximation if full history not available)
        ath = float(df_1d['close'].max())
        if ath > 0:
            pct_down_ath = float(((last_close - ath) / ath) * 100.0)

    # Prepare sparkline series (2 weeks of 15m closes)
    spark_series = closes_15.tail(14 * 24 * 4)  # up to last 2 weeks (96 * 14)

    row = {
        "symbol": symbol,
        "class": cls,
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
    # Write as: Row per symbol -> [symbol, v1, v2, v3, ...]
    header = ["symbol"]
    # Determine max series length to layout columns consistently
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
    # Order columns
    columns = [
        "symbol", "class", "%down_from_ATH", "PL%_1d", "PL%_7d", "PL%_14d",
        "volume_24h", "RSI14_15m", "MA60_15m", "MA240_15m", "MACD_15m", "MACDsig_15m", "sparkline"
    ]

    values = [columns]
    for r in rows:
        base = [r.get(c, "") for c in columns[:-1]]  # all except sparkline
        # sparkline formula references chartData row for this symbol
        # =SPARKLINE(INDEX('Alpaca-Screener-chartData'!B:ZZ, MATCH(A2, 'Alpaca-Screener-chartData'!$A:$A, 0), 0))
        formula = (
            f"=IFERROR(SPARKLINE(INDEX('" + CHARTDATA_TAB + "'!B:ZZ, "
            f"MATCH(A{{row}}, '" + CHARTDATA_TAB + "'!$A:$A, 0), 0)), \"\")"
        )
        base.append(formula)
        values.append(base)

    ws_main.clear()
    ws_main.update('A1', values)

    # Now convert the {row} placeholders into actual row numbers for the sparkline formulas
    total_rows = len(values)
    spark_col = len(values[0])
    cell_updates = []
    for i in range(2, total_rows + 1):  # starting from row 2 (excluding header)
        cell_updates.append({
            'range': gspread.utils.rowcol_to_a1(i, spark_col),
            'values': [[values[i-1][-1].replace('{row}', str(i))]]
        })

    # Batch update sparkline column formulas
    ws_main.batch_update([{'range': u['range'], 'values': u['values']} for u in cell_updates])

# -------------------------
# Orchestration
# -------------------------
def run_once():
    ensure_kraken_connectivity()

    gc = get_gspread_client()
    sh = open_or_create_sheet(gc)
    ws_main = ensure_worksheet(sh, SCREENER_TAB, rows=1000, cols=30)
    ws_chart = ensure_worksheet(sh, CHARTDATA_TAB, rows=1000, cols=500)

    assets = list_alpaca_tradable_assets()
    print(f"Found {len(assets)} Alpaca tradable assets")

    rows: List[Dict] = []
    spark_map: Dict[str, pd.Series] = {}

    # Process in batches to respect data API limits
    for i in range(0, len(assets), BATCH_SIZE):
        batch = assets[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1} containing {len(batch)} assets ...")
        for a in batch:
            sym, cls = a['symbol'], a['class']
            try:
                row, spark = compute_metrics_for_symbol(sym, cls)
            except Exception as e:
                print(f"[WARN] compute failed for {sym}: {e}")
                row, spark = None, None
            if row is not None:
                rows.append(row)
                spark_map[sym] = spark
        time.sleep(SLEEP_BETWEEN_BATCHES)

    # Sort rows by symbol for stable output
    rows.sort(key=lambda r: r.get('symbol', ''))

    write_chartdata(ws_chart, spark_map)
    write_screener(ws_main, rows)

if __name__ == "__main__":
    print("Starting Alpaca ➜ Google Sheets screener loop…")
    refresh_seconds = max(60, REFRESH_MINUTES * 60)
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[ERROR] run_once failed: {e}")
        print(f"Done. Sleeping for {REFRESH_MINUTES} minutes…")
        time.sleep(refresh_seconds)
