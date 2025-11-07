import os
import json
import time
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

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
# Env / Settings
# -------------------------
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Trading Log")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON", "")

# Throttling & batching
APCA_MAX_RPM = int(os.environ.get("APCA_MAX_RPM", 120))        # requests/min cap to stay below Alpaca limit
REQ_SLEEP = max(0.0, 60.0 / max(1, APCA_MAX_RPM))              # seconds to sleep per request
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 25))
SLEEP_BETWEEN_BATCHES = float(os.environ.get("SLEEP_BETWEEN_BATCHES", 0.5))

# Intraday requirements
# 2 weeks of 15m bars = 14 * 24 * 4 = 1344
MIN_15M_BARS = int(os.environ.get("MIN_15M_BARS", 1344))
# Force allow daily-only rows so you always get output even if 15m is thin
REQUIRE_FULL_DATA = False

REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", 30))
TZ = pytz.UTC

SCREENER_TAB = "Alpaca-Screener"
CHARTDATA_TAB = "Alpaca-Screener-chartData"

# Our owned columns A–M (don’t touch N+)
SCREENER_COLUMNS = [
    "symbol", "class", "%down_from_ATH", "PL%_1d", "PL%_7d", "PL%_14d",
    "volume_24h", "RSI14_15m", "MA60_15m", "MA240_15m", "MACD_15m", "MACDsig_15m", "sparkline"
]
SCREENER_LAST_COL = "M"  # A..M

# Exchanges to include
ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "BATS", "NYSEARCA"}

# NEW: symbol sampling controls
MAX_SYMBOLS = os.environ.get("MAX_SYMBOLS")                 # e.g., "500"
SYMBOLS_OFFSET = int(os.environ.get("SYMBOLS_OFFSET", 0))   # e.g., 2000
SHUFFLE_SYMBOLS = os.environ.get("SHUFFLE_SYMBOLS", "false").lower() == "true"
SAMPLE_SEED = os.environ.get("SAMPLE_SEED")                 # e.g., "42"

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
# Sheets helpers
# -------------------------
def get_gspread_client():
    if not GOOGLE_CREDS_JSON:
        raise RuntimeError("GOOGLE_CREDS_JSON env var is empty. Provide service account JSON.")
    info = json.loads(GOOGLE_CREDS_JSON)
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
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
# Assets list (REST; equities only, allowed exchanges)
# -------------------------
def list_alpaca_tradable_equities() -> List[Dict]:
    url = APCA_API_BASE_URL.rstrip('/') + "/v2/assets"
    headers = {"APCA-API-KEY-ID": APCA_API_KEY_ID or "", "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY or ""}
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        assets = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch assets via HTTP: {e}")
        return []

    unique, seen = [], set()
    for a in assets:
        symbol = a.get("symbol")
        if not symbol or symbol in seen:
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
        unique.append({"symbol": symbol})
        seen.add(symbol)
    return unique

# -------------------------
# Rate-limited call wrapper
# -------------------------
def pace_request():
    if REQ_SLEEP > 0:
        time.sleep(REQ_SLEEP)

# -------------------------
# Batched, paginated data fetching (IEX only)
# -------------------------
class TransientDataError(Exception): pass

def _bars_request(symbols: List[str], tf: TimeFrame, start: datetime, end: datetime, page_token: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start,
        end=end,
        feed=DataFeed.IEX,  # IEX only (SIP disabled)
        page_token=page_token
    )
    resp = stock_data_client.get_stock_bars(req)  # BarsResponse
    df = getattr(resp, "df", None)
    next_token = getattr(resp, "next_page_token", None)
    return (df if df is not None else pd.DataFrame()), next_token

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(TransientDataError)
)
def fetch_bars_batch_paginated(symbols: List[str], tf: TimeFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch all pages for (symbols, timeframe, start-end)."""
    all_frames = []
    token = None
    page = 0
    while True:
        page += 1
        try:
            df, token = _bars_request(symbols, tf, start, end, token)
            if page % 5 == 0:
                print(f"[INFO]  fetched page {page} for {tf} with {0 if df is None else len(df)} rows")
            pace_request()
        except Exception as e:
            msg = str(e).lower()
            if any(m in msg for m in ["timeout", "temporarily", "rate limit", "429", "connection reset", "502", "503", "504"]):
                raise TransientDataError(msg)
            raise
        if df is not None and not df.empty:
            all_frames.append(df)
        if not token:
            break
    if not all_frames:
        return pd.DataFrame()
    out = pd.concat(all_frames)
    try:
        out = out.sort_index()
    except Exception:
        pass
    return out

def slice_symbol_df(df: Optional[pd.DataFrame], symbol: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if isinstance(df.index, pd.MultiIndex):
        try:
            sdf = df.xs(symbol, level='symbol')
        except KeyError:
            return None
    else:
        sdf = df[df.get('symbol', '') == symbol] if 'symbol' in df.columns else df
    return sdf.sort_index() if sdf is not None and not sdf.empty else None

# -------------------------
# Per-symbol computation (daily always; 15m if available)
# -------------------------
def compute_row(symbol: str, df_15_batch: Optional[pd.DataFrame], df_1d_batch: Optional[pd.DataFrame]) -> Optional[Tuple[Dict, pd.Series]]:
    # Daily metrics
    s1d = slice_symbol_df(df_1d_batch, symbol)

    pl_1d = pl_7d = pl_14d = pct_down_ath = np.nan
    if s1d is not None and not s1d.empty:
        last_close = s1d['close'].iloc[-1]
        def pct_change_n_days(n):
            cutoff = s1d.index.max() - pd.Timedelta(days=n)
            prior = s1d[s1d.index <= cutoff]
            if prior.empty: return np.nan
            ref = prior['close'].iloc[-1]
            return float(((last_close - ref) / ref) * 100.0) if ref else np.nan
        pl_1d  = pct_change_n_days(1)
        pl_7d  = pct_change_n_days(7)
        pl_14d = pct_change_n_days(14)
        ath = float(s1d['close'].max())
        if ath > 0:
            pct_down_ath = float(((last_close - ath) / ath) * 100.0)

    # Intraday metrics
    s15 = slice_symbol_df(df_15_batch, symbol)
    rsi14 = ma60_last = ma240_last = macd_last = macd_signal_last = vol_24h = np.nan
    spark_series = pd.Series(dtype=float)
    if s15 is not None and len(s15) >= MIN_15M_BARS:
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
        vol_24h = float(volumes_15.tail(96).sum(skipna=True)) if len(volumes_15) else np.nan
        spark_series = closes_15.tail(14 * 24 * 4)

    # Build row even if some fields are blank (we're not requiring full data)
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
# Sheets writing (A–M only; formulas parsed; correct call order)
# -------------------------
def write_chartdata(ws_chart: gspread.Worksheet, all_spark: Dict[str, pd.Series]):
    header = ["symbol"]
    max_len = max((len(s) for s in all_spark.values() if s is not None), default=0)
    header += [f"p{i+1}" for i in range(max_len)]
    values = [header]
    for sym, series in all_spark.items():
        row = [sym] + (list(map(lambda x: float(x) if pd.notna(x) else "", series.values)) if series is not None else [])
        values.append(row)
    ws_chart.update(values, 'A1', raw=False)

def write_screener_partial(ws_main: gspread.Worksheet, rows: List[Dict]):
    header = [c for c in SCREENER_COLUMNS]
    ws_main.update([header], f"A1:{SCREENER_LAST_COL}1", raw=False)

    out = []
    for i, r in enumerate(rows, start=2):
        base = [r.get(c, "") for c in SCREENER_COLUMNS[:-1]]  # without sparkline
        formula = f"=IFERROR(SPARKLINE(INDEX('{CHARTDATA_TAB}'!B:ZZ, MATCH(A{i}, '{CHARTDATA_TAB}'!$A:$A, 0), 0)), \"\")"
        base.append(formula)
        out.append(base)

    if out:
        ws_main.update(out, f"A2:{SCREENER_LAST_COL}{len(out)+1}", raw=False)

# -------------------------
# Orchestration
# -------------------------
def run_once():
    print(f"[BOOT] APCA_MAX_RPM={APCA_MAX_RPM} (sleep/request {REQ_SLEEP:.3f}s), BATCH_SIZE={BATCH_SIZE}, "
          f"SLEEP_BETWEEN_BATCHES={SLEEP_BETWEEN_BATCHES}, MIN_15M_BARS={MIN_15M_BARS}, "
          f"REQUIRE_FULL_DATA={REQUIRE_FULL_DATA}, SHUFFLE_SYMBOLS={SHUFFLE_SYMBOLS}, "
          f"SYMBOLS_OFFSET={SYMBOLS_OFFSET}, MAX_SYMBOLS={MAX_SYMBOLS}, SAMPLE_SEED={SAMPLE_SEED}")
    gc = get_gspread_client()
    sh = open_or_create_sheet(gc)
    ws_main = ensure_worksheet(sh, SCREENER_TAB, rows=5000, cols=200)
    ws_chart = ensure_worksheet(sh, CHARTDATA_TAB, rows=5000, cols=2000)

    assets = list_alpaca_tradable_equities()
    symbols = [a["symbol"] for a in assets]

    # Optional shuffle
    if SHUFFLE_SYMBOLS:
        rng = np.random.default_rng(int(SAMPLE_SEED) if SAMPLE_SEED else None)
        rng.shuffle(symbols)

    # Optional offset
    if SYMBOLS_OFFSET:
        symbols = symbols[SYMBOLS_OFFSET:]

    # Optional cap
    if MAX_SYMBOLS:
        try:
            cap = int(MAX_SYMBOLS)
            if cap > 0:
                symbols = symbols[:cap]
        except ValueError:
            pass

    print(f"[INFO] Symbols after exchange filter + sampling: {len(symbols)}")

    rows: List[Dict] = []
    spark_map: Dict[str, pd.Series] = {}

    end = utcnow()
    start_15m = end - timedelta(days=14, minutes=5)
    start_1d = end - timedelta(days=365*5)

    kept = 0
    for i in range(0, len(symbols), BATCH_SIZE):
        batch_syms = symbols[i:i+BATCH_SIZE]
        print(f"[INFO] Batch {i//BATCH_SIZE + 1}: fetching bars for {len(batch_syms)} symbols…")

        df_15 = pd.DataFrame()
        df_1d = pd.DataFrame()
        try:
            df_15 = fetch_bars_batch_paginated(batch_syms, TimeFrame(15, TimeFrameUnit.Minute), start_15m, end)
        except Exception as e:
            print(f"[WARN] 15m batch fetch failed: {e}")

        try:
            df_1d = fetch_bars_batch_paginated(batch_syms, TimeFrame.Day, start_1d, end)
        except Exception as e:
            print(f"[WARN] 1D batch fetch failed: {e}")

        for sym in batch_syms:
            try:
                row, spark = compute_row(sym, df_15, df_1d)
                rows.append(row)
                spark_map[sym] = spark
                kept += 1
            except Exception as e:
                print(f"[WARN] compute failed for {sym}: {e}")

        print(f"[INFO] Batch cumulative kept: {kept}")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    rows.sort(key=lambda r: r.get('symbol', ''))
    print(f"[INFO] Writing chart data for {len(spark_map)} symbols and screener rows: {len(rows)}")
    write_chartdata(ws_chart, spark_map)
    write_screener_partial(ws_main, rows)
    print(f"[INFO] Write complete.")

if __name__ == "__main__":
    print("Starting Alpaca ➜ Google Sheets screener loop (IEX-only, paginated, A–M only, daily allowed, sampling)…")
    refresh_seconds = max(60, REFRESH_MINUTES * 60)
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[ERROR] run_once failed: {e}")
        print(f"Done. Sleeping for {REFRESH_MINUTES} minutes…")
        time.sleep(refresh_seconds)
