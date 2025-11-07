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
APCA_MAX_RPM = int(os.environ.get("APCA_MAX_RPM", 120))          # requests/min cap
REQ_SLEEP = max(0.0, 60.0 / max(1, APCA_MAX_RPM))                # seconds between requests
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 25))
SLEEP_BETWEEN_BATCHES = float(os.environ.get("SLEEP_BETWEEN_BATCHES", 0.5))

# Intraday (1-hour bars)
# 2 weeks of 1h bars ≈ 14 * 24 = 336
MIN_1H_BARS = int(os.environ.get("MIN_1H_BARS", 320))            # near-full two weeks
REQUIRE_FULL_DATA = os.environ.get("REQUIRE_FULL_DATA", "false").lower() == "true"  # default: allow daily-only rows

REFRESH_MINUTES = int(os.environ.get("REFRESH_MINUTES", 30))
TZ = pytz.UTC

SCREENER_TAB = "Alpaca-Screener"
CHARTDATA_TAB = "Alpaca-Screener-chartData"

# Our owned columns A–M (don’t touch N+)
SCREENER_COLUMNS = [
    "symbol", "class", "%down_from_ATH", "PL%_1d", "PL%_7d", "PL%_14d",
    "volume_24h", "RSI14_1h", "MA15_1h", "MA60_1h", "MACD_1h", "MACDsig_1h", "sparkline"
]
SCREENER_LAST_COL = "M"  # A..M

# Exchanges to include
ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "BATS", "NYSEARCA"}

# Symbol sampling controls
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
# Symbol class helpers
# -------------------------
BAD_SUFFIXES = ("W", "WS", "U", "R")  # warrants, units, rights (coarse)
BAD_NAME_KEYWORDS = (
    "preferred", "warrant", "unit", "right", "note", "bond", "debenture", "depositary share", "perp"
)
ETF_NAME_KEYWORDS = (" etf", " index fund", "index etf", "exchange-traded fund")

def is_symbol_common_stock(sym: str, name: str) -> bool:
    s = sym.upper()
    if any(ch in s for ch in ('.', '-', '/')):  # series, preferred, units etc.
        return False
    if s.endswith(BAD_SUFFIXES):
        return False
    # Exclude by name clues
    n = (name or "").lower()
    if any(k in n for k in BAD_NAME_KEYWORDS):
        return False
    return True

def is_symbol_etf(sym: str, name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ETF_NAME_KEYWORDS)

def is_symbol_eligible(sym: str, name: str) -> bool:
    # Allow common stocks or ETFs; exclude everything else.
    return is_symbol_common_stock(sym, name) or is_symbol_etf(sym, name)

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
# Assets list (REST; equities only, allowed exchanges) → keep only stocks & ETFs
# -------------------------
def list_alpaca_tradable_stocks_and_etfs() -> List[Dict]:
    url = APCA_API_BASE_URL.rstrip('/') + "/v2/assets"
    headers = {"APCA-API-KEY-ID": APCA_API_KEY_ID or "", "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY or ""}
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        assets = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch assets via HTTP: {e}")
        return []

    kept, seen = [], set()
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

        name = a.get("name") or ""
        if not is_symbol_eligible(symbol, name):
            continue

        kept.append({"symbol": symbol, "name": name})
        seen.add(symbol)

    print(f"[INFO] Eligible stocks/ETFs after filters: {len(kept)}")
    return kept

# -------------------------
# Rate-limited call wrapper
# -------------------------
def pace_request():
    if REQ_SLEEP > 0:
        time.sleep(REQ_SLEEP)

# -------------------------
# Data fetchers
# -------------------------
class TransientDataError(Exception): pass

def _bars_request_multi(symbols: List[str], tf: TimeFrame, start: datetime, end: datetime, page_token: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        page_token=page_token
    )
    resp = stock_data_client.get_stock_bars(req)
    df = getattr(resp, "df", None)
    next_token = getattr(resp, "next_page_token", None)
    return (df if df is not None else pd.DataFrame()), next_token

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(TransientDataError))
def fetch_daily_batch(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    all_frames = []
    token = None
    page = 0
    while True:
        page += 1
        try:
            df, token = _bars_request_multi(symbols, TimeFrame.Day, start, end, token)
            if page % 5 == 0:
                print(f"[INFO]  daily page {page} rows={0 if df is None else len(df)}")
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

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type(TransientDataError))
def fetch_1h_one(symbol: str, bars_needed: int = 336) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        limit=bars_needed,
        feed=DataFeed.IEX
    )
    try:
        resp = stock_data_client.get_stock_bars(req)
        df = getattr(resp, "df", None)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level='symbol')
            except KeyError:
                return pd.DataFrame()
        elif 'symbol' in df.columns:
            df = df[df['symbol'] == symbol]
        df = df.sort_index()
        return df
    except Exception as e:
        msg = str(e).lower()
        if any(m in msg for m in ["timeout", "temporarily", "rate limit", "429", "connection reset", "502", "503", "504"]):
            raise TransientDataError(msg)
        return pd.DataFrame()
    finally:
        pace_request()

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
# Per-symbol computation (daily + 1h intraday)
# -------------------------
def compute_row(symbol: str, df_1d_batch: Optional[pd.DataFrame], df_1h_one: Optional[pd.DataFrame]) -> Optional[Tuple[Dict, pd.Series]]:
    # Daily metrics
    s1d = slice_symbol_df(df_1d_batch, symbol)

    pl_1d = pl_7d = pl_14d = pct_down_ath = np.nan
    if s1d is not None and not s1d.empty:
        s1d = s1d.sort_index()
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

    # 1-hour intraday metrics
    rsi14 = ma15_last = ma60_last = macd_last = macd_signal_last = vol_24h = np.nan
    spark_series = pd.Series(dtype=float)
    if df_1h_one is not None and not df_1h_one.empty and len(df_1h_one) >= MIN_1H_BARS:
        closes_1h = df_1h_one['close']
        volumes_1h = df_1h_one['volume'] if 'volume' in df_1h_one.columns else pd.Series(index=df_1h_one.index, data=np.nan)
        rsi14_series = rsi(closes_1h, 14)
        ma15 = closes_1h.rolling(window=15, min_periods=1).mean()   # MA15 @ 1h
        ma60 = closes_1h.rolling(window=60, min_periods=1).mean()   # MA60 @ 1h
        macd_line, signal_line = macd(closes_1h)
        rsi14 = float(rsi14_series.iloc[-1]) if len(rsi14_series) else np.nan
        ma15_last = float(ma15.iloc[-1]) if len(ma15) else np.nan
        ma60_last = float(ma60.iloc[-1]) if len(ma60) else np.nan
        macd_last = float(macd_line.iloc[-1]) if len(macd_line) else np.nan
        macd_signal_last = float(signal_line.iloc[-1]) if len(signal_line) else np.nan
        # last 24 hours of 1h volume
        vol_24h = float(volumes_1h.tail(24).sum(skipna=True)) if len(volumes_1h) else np.nan
        # sparkline series = last 2 weeks of 1h closes (≈336 values)
        spark_series = closes_1h.tail(14 * 24)

    # Require full?
    if REQUIRE_FULL_DATA:
        need_daily = [pct_down_ath, pl_1d, pl_7d, pl_14d]
        need_intraday = [rsi14, ma15_last, ma60_last, macd_last, macd_signal_last]
        if any(math.isnan(x) for x in need_daily) or any(math.isnan(x) for x in need_intraday) or len(spark_series) < (14*24):
            return None

    row = {
        "symbol": symbol,
        "class": "us_equity",
        "%down_from_ATH": round(pct_down_ath, 4) if not math.isnan(pct_down_ath) else "",
        "PL%_1d": round(pl_1d, 4) if not math.isnan(pl_1d) else "",
        "PL%_7d": round(pl_7d, 4) if not math.isnan(pl_7d) else "",
        "PL%_14d": round(pl_14d, 4) if not math.isnan(pl_14d) else "",
        "volume_24h": round(vol_24h, 4) if not math.isnan(vol_24h) else "",
        "RSI14_1h": round(rsi14, 4) if not math.isnan(rsi14) else "",
        "MA15_1h": round(ma15_last, 6) if not math.isnan(ma15_last) else "",
        "MA60_1h": round(ma60_last, 6) if not math.isnan(ma60_last) else "",
        "MACD_1h": round(macd_last, 6) if not math.isnan(macd_last) else "",
        "MACDsig_1h": round(macd_signal_last, 6) if not math.isnan(macd_signal_last) else "",
    }
    return row, spark_series

# -------------------------
# Sheets writing (A–M only; formulas parsed)
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
          f"SLEEP_BETWEEN_BATCHES={SLEEP_BETWEEN_BATCHES}, MIN_1H_BARS={MIN_1H_BARS}, "
          f"REQUIRE_FULL_DATA={REQUIRE_FULL_DATA}, SHUFFLE_SYMBOLS={SHUFFLE_SYMBOLS}, "
          f"SYMBOLS_OFFSET={SYMBOLS_OFFSET}, MAX_SYMBOLS={MAX_SYMBOLS}, SAMPLE_SEED={SAMPLE_SEED}")

    gc = get_gspread_client()
    sh = open_or_create_sheet(gc)
    ws_main = ensure_worksheet(sh, SCREENER_TAB, rows=5000, cols=200)
    ws_chart = ensure_worksheet(sh, CHARTDATA_TAB, rows=5000, cols=2000)

    assets = list_alpaca_tradable_stocks_and_etfs()
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

    print(f"[INFO] Symbols after stock/ETF filter + sampling: {len(symbols)}")

    rows: List[Dict] = []
    spark_map: Dict[str, pd.Series] = {}

    end = utcnow()
    start_1d = end - timedelta(days=365*5)

    kept = 0
    dropped_full_required = 0
    dropped_no_daily = 0
    dropped_no_1h = 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch_syms = symbols[i:i+BATCH_SIZE]
        print(f"[INFO] Batch {i//BATCH_SIZE + 1}: daily for {len(batch_syms)}; 1h per-symbol …")

        # Fetch daily for the whole batch (fast)
        df_1d = pd.DataFrame()
        try:
            df_1d = fetch_daily_batch(batch_syms, start_1d, end)
        except Exception as e:
            print(f"[WARN] 1D batch fetch failed: {e}")

        # Per-symbol 1h
        for sym in batch_syms:
            try:
                df_1h = fetch_1h_one(sym, bars_needed=336)
                res = compute_row(sym, df_1d, df_1h)
                if res is None:
                    has_daily = slice_symbol_df(df_1d, sym) is not None and not slice_symbol_df(df_1d, sym).empty
                    has_1h = df_1h is not None and len(df_1h) >= MIN_1H_BARS
                    if not has_daily:
                        dropped_no_daily += 1
                    elif not has_1h:
                        dropped_no_1h += 1
                    else:
                        dropped_full_required += 1
                    continue

                row, spark = res
                rows.append(row)
                spark_map[sym] = spark
                kept += 1
            except Exception as e:
                print(f"[WARN] compute failed for {sym}: {e}")

        print(f"[INFO] Batch cumulative kept={kept} / dropped_no_daily={dropped_no_daily} "
              f"/ dropped_no_1h={dropped_no_1h} / dropped_full_required={dropped_full_required}")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    rows.sort(key=lambda r: r.get('symbol', ''))
    print(f"[INFO] Writing chart data for {len(spark_map)} symbols and screener rows: {len(rows)}")
    write_chartdata(ws_chart, spark_map)
    write_screener_partial(ws_main, rows)
    print(f"[INFO] Write complete.")

if __name__ == "__main__":
    print("Starting Alpaca ➜ Google Sheets screener loop (IEX-only, stocks+ETFs, daily batched + 1h per-symbol, A–M only)…")
    refresh_seconds = max(60, REFRESH_MINUTES * 60)
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[ERROR] run_once failed: {e}")
        print(f"Done. Sleeping for {REFRESH_MINUTES} minutes…")
        time.sleep(refresh_seconds)
