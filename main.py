#!/usr/bin/env python3
import os, sys, json, time, math, random, logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timezone, timedelta

import requests
import gspread
from google.oauth2.service_account import Credentials
from gspread_formatting import (
    format_cell_ranges, CellFormat, NumberFormat, set_frozen
)

# =========================
# Config (per your request)
# =========================
APCA_API_KEY_ID     = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")

GOOGLE_CREDS_JSON   = os.getenv("GOOGLE_CREDS_JSON", "")
GOOGLE_SHEET_NAME   = os.getenv("GOOGLE_SHEET_NAME", "Trading Log")
MAX_SYMBOLS         = int(os.getenv("MAX_SYMBOLS", "1000"))

# Optional tunables
HTTP_TIMEOUT        = int(os.getenv("HTTP_TIMEOUT", "15"))
APCA_API_BASE       = os.getenv("APCA_API_BASE", "https://api.alpaca.markets")        # Trading API
APCA_DATA_BASE      = os.getenv("APCA_DATA_BASE", "https://data.alpaca.markets")      # Data API v2
ALPACA_FEED         = os.getenv("ALPACA_FEED", "iex")                                  # "iex" or "sip"
LOOKBACK_DAYS_1D    = int(os.getenv("LOOKBACK_DAYS_1D", "450"))                        # for ATH/returns/volume
LOOKBACK_DAYS_15M   = int(os.getenv("LOOKBACK_DAYS_15M", "60"))                        # for 15m indicators
SPARK_LEN           = int(os.getenv("SPARK_LEN", "100"))                               # # of 15m closes in sparkline

ASSETS_WS           = os.getenv("ASSETS_WS", "Alpaca-Screener")  # target sheet name
SPARK_WS            = os.getenv("SPARK_WS", "spark_data")       # sparkline backing data

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

UA = {"User-Agent": "Mozilla/5.0 (compatible; AlpacaAssetsBot/1.2; +https://example.org/bot)"}

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =========================
# HTTP session with retries
# =========================
_session = requests.Session()
try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    _session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20))
    _session.headers.update(UA)
except Exception:
    pass

# simple per-host pacing
_next_ok_at: Dict[str, float] = {}
def _pace(url: str, base_delay=0.12):
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    now = time.time()
    wait_until = _next_ok_at.get(host, now)
    if wait_until > now:
        time.sleep(wait_until - now)
    _next_ok_at[host] = time.time() + base_delay * random.uniform(0.9, 1.3)

def _sleep_with_jitter(seconds: float):
    time.sleep(min(seconds, 30.0) * random.uniform(0.85, 1.15))

def fetch_json_with_retries(url: str, *, params=None, headers=None, timeout=HTTP_TIMEOUT,
                            retries=4, backoff_base=0.7) -> Dict:
    last_exc = None
    for attempt in range(retries):
        try:
            _pace(url)
            r = _session.get(url, params=params, headers=headers or UA, timeout=timeout)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                delay = float(ra) if ra else backoff_base * (2 ** attempt)
                logging.info(f"429 at {url} — sleeping {delay:.2f}s")
                _sleep_with_jitter(delay)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            _sleep_with_jitter(backoff_base * (2 ** attempt))
    raise last_exc

# =========================
# Sheets helpers
# =========================
def get_client():
    if not GOOGLE_CREDS_JSON:
        logging.error("GOOGLE_CREDS_JSON env var is missing.")
        sys.exit(1)
    info = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def open_sheet(gc):
    return gc.open(GOOGLE_SHEET_NAME)

def ensure_worksheet(sh, title: str, rows: int = 1000, cols: int = 10):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)

def ensure_size(ws, min_rows: int, min_cols: int):
    """Grow the worksheet if it's smaller than needed."""
    try:
        rows = getattr(ws, "row_count", None)
        cols = getattr(ws, "col_count", None)
        if rows is None or cols is None:
            meta = ws.spreadsheet.fetch_sheet_metadata()
            for sht in meta.get("sheets", []):
                props = sht.get("properties", {})
                if props.get("title") == ws.title:
                    gp = props.get("gridProperties", {}) or {}
                    rows = gp.get("rowCount", 1000)
                    cols = gp.get("columnCount", 26)
                    break
        need_rows = max(int(rows or 0), int(min_rows))
        need_cols = max(int(cols or 0), int(min_cols))
        if need_rows > (rows or 0) or need_cols > (cols or 0):
            ws.resize(rows=need_rows, cols=need_cols)
    except Exception:
        # If resizing fails, writes may error if truly too small
        pass

def _json_safe_cell(v):
    if v is None:
        return ""
    try:
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return ""
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            return v
        if hasattr(v, "__float__"):
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return ""
            return f
    except Exception:
        pass
    return str(v)

def _json_safe_rows(rows: List[List[object]]) -> List[List[object]]:
    return [[_json_safe_cell(c) for c in row] for row in rows]

def replace_sheet(ws, rows: List[List[object]], header: List[str], *, value_input_option: str = "RAW"):
    # Determine required size (header + data) and ensure the grid is big enough
    data_rows = len(rows) if rows else 0
    data_cols = max(len(header), max((len(r) for r in rows), default=0))
    ensure_size(ws, min_rows=data_rows + 2, min_cols=data_cols + 2)

    # Clear and write header + data
    ws.batch_clear(["A:ZZ"])

    # New signature: values first, then range_name (or named args)
    ws.update(values=[header], range_name="A1", value_input_option=value_input_option)

    if rows:
        rows = _json_safe_rows(rows)
        CHUNK = 1000
        start_row = 2
        for i in range(0, len(rows), CHUNK):
            block = rows[i:i+CHUNK]
            ws.update(values=block, range_name=f"A{start_row}", value_input_option=value_input_option)
            start_row += len(block)

    logging.info(f"Wrote {len(rows) if rows else 0} rows to '{ws.title}'.")

# =========================
# Indicators
# =========================
def sma(values: List[float], window: int) -> float:
    if len(values) < window:
        return float("nan")
    return round(sum(values[-window:]) / window, 4)

def rsi14(values: List[float]) -> float:
    period = 14
    if len(values) < period + 1:
        return float("nan")
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        ch = values[i] - values[i-1]
        if ch > 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = gains / losses
    return round(100 - (100 / (1 + rs)), 2)

def macd_last(values: List[float]) -> Tuple[float, float, float]:
    """Standard MACD(12,26,9) over closes; returns (macd, signal, hist) for the last bar."""
    if len(values) < 26 + 9:
        return (float("nan"), float("nan"), float("nan"))
    k12, k26 = 2/(12+1), 2/(26+1)
    # seed EMAs with SMA
    ema12 = sum(values[:12]) / 12
    ema26 = sum(values[:26]) / 26
    # advance ema12 across values[12:26]
    for v in values[12:26]:
        ema12 = v * k12 + ema12 * (1 - k12)
    macd_series = []
    for v in values[26:]:
        ema12 = v * k12 + ema12 * (1 - k12)
        ema26 = v * k26 + ema26 * (1 - k26)
        macd_series.append(ema12 - ema26)
    if len(macd_series) < 9:
        return (float("nan"), float("nan"), float("nan"))
    # signal EMA(9) over MACD series
    k9 = 2/(9+1)
    sig = sum(macd_series[:9]) / 9
    for m in macd_series[9:]:
        sig = m * k9 + sig * (1 - k9)
    macd_val = macd_series[-1]
    hist = macd_val - sig
    return (round(macd_val, 4), round(sig, 4), round(hist, 4))

# =========================
# Alpaca helpers
# =========================
def _apca_enabled() -> bool:
    return bool(APCA_API_KEY_ID and APCA_API_SECRET_KEY)

def _apca_headers() -> Dict[str, str]:
    return {
        **UA,
        "APCA-API-KEY-ID": APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY,
        "Accept": "application/json",
    }

def _iso_utc_days_ago(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")

def get_apca_tradable_assets(limit: int = MAX_SYMBOLS) -> List[str]:
    """
    Pull all tradable, active US equities from Alpaca Trading API.
    """
    if not _apca_enabled():
        logging.error("[APCA] API keys missing; cannot fetch assets.")
        return []
    url = f"{APCA_API_BASE}/v2/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    try:
        data = fetch_json_with_retries(url, params=params, headers=_apca_headers(), timeout=20)
        syms = []
        for a in data or []:
            try:
                if a.get("tradable", False):
                    s = (a.get("symbol") or "").strip().upper()
                    if 1 <= len(s) <= 12:
                        syms.append(s)
            except Exception:
                continue
        syms = sorted(set(syms))
        if limit and len(syms) > limit:
            syms = syms[:limit]
        logging.info(f"[APCA] Tradable assets: {len(syms)} (cap={limit})")
        return syms
    except Exception as e:
        logging.warning(f"[APCA] Error fetching assets: {e}")
        return []

def get_bars(symbol: str, timeframe: str, start_iso: str, limit: int = 5000) -> List[Dict]:
    """
    Generic bars fetcher from Data API v2. Returns list of bars (dicts with c,h,l,o,v,t).
    timeframe examples: '1Day', '15Min'
    """
    url = f"{APCA_DATA_BASE}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "adjustment": "raw",
        "feed": ALPACA_FEED,
        "start": start_iso,
        "limit": str(limit),
    }
    try:
        data = fetch_json_with_retries(url, params=params, headers=_apca_headers(), timeout=20)
        return data.get("bars", []) or []
    except Exception as e:
        logging.info(f"[APCA] {symbol} bars({timeframe}) error: {e}")
        return []

# =========================
# Metrics per asset (strict completeness)
# =========================
def _is_finite(x) -> bool:
    return (x is not None) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def compute_asset_metrics(symbol: str) -> Optional[Tuple[list, list]]:
    """
    Returns:
      - metrics row for Assets sheet
      - sparkline row for Spark Data sheet (symbol followed by recent 15m closes)
    Returns None if ANY required metric cannot be computed (strict completeness).
    """
    # --- Daily bars: ATH, returns, 24h volume (use last daily volume)
    daily_bars = get_bars(symbol, "1Day", _iso_utc_days_ago(LOOKBACK_DAYS_1D), limit=2000)
    d_close = [float(b.get("c")) for b in daily_bars if "c" in b]
    d_high  = [float(b.get("h")) for b in daily_bars if "h" in b]
    d_vol   = [float(b.get("v")) for b in daily_bars if "v" in b]

    if not d_close or not d_high or not d_vol:
        return None

    last_price = d_close[-1]
    ath = max(d_high)
    if not _is_finite(last_price) or not _is_finite(ath) or ath <= 0:
        return None
    down_from_ath_pct = round((ath - last_price) / ath * 100.0, 4)

    # returns vs N days ago close if available
    if len(d_close) < 15:
        return None

    def pct(now: float, then: float) -> float:
        if then and then > 0:
            return round((now / then - 1.0) * 100.0, 4)
        return float("nan")

    r1  = pct(last_price, d_close[-2])
    r7  = pct(last_price, d_close[-8])
    r14 = pct(last_price, d_close[-15])
    vol_24h = d_vol[-1]

    # --- 15m bars: RSI14, MA60, MA240, MACD + spark
    m15_bars = get_bars(symbol, "15Min", _iso_utc_days_ago(LOOKBACK_DAYS_15M), limit=5000)
    m15_close = [float(b.get("c")) for b in m15_bars if "c" in b]

    if len(m15_close) < 240:  # need enough for MA240 & MACD signal (strict)
        return None

    rsi = rsi14(m15_close)
    ma60 = sma(m15_close, 60)
    ma240 = sma(m15_close, 240)
    macd, macd_sig, macd_hist = macd_last(m15_close)

    # Validate all required metrics
    required = [last_price, down_from_ath_pct, r1, r7, r14, vol_24h, rsi, ma60, ma240, macd, macd_sig, macd_hist]
    if not all(_is_finite(x) for x in required):
        return None

    # Sparkline series: last up to SPARK_LEN closes
    spark_series = m15_close[-SPARK_LEN:]
    if not spark_series:
        return None

    metrics_row = [
        symbol,
        last_price,
        down_from_ath_pct,
        r1, r7, r14,
        vol_24h,
        rsi, ma60, ma240,
        macd, macd_sig, macd_hist,
        ""  # placeholder for SPARKLINE formula
    ]
    spark_row = [symbol] + spark_series
    return (metrics_row, spark_row)

# =========================
# Pipeline: write sheets
# =========================
def run_assets(gc):
    if not _apca_enabled():
        logging.error("[APCA] Disabled — set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        return

    sh = open_sheet(gc)
    # Create or open the sheets; exact size will be adjusted dynamically in replace_sheet()
    ws_assets = ensure_worksheet(sh, ASSETS_WS)
    ws_spark  = ensure_worksheet(sh, SPARK_WS)

    # Universe
    symbols = get_apca_tradable_assets(MAX_SYMBOLS)
    if not symbols:
        logging.info("[Assets] No symbols returned.")
        return

    assets_rows: List[List[object]] = []
    spark_rows:  List[List[object]] = []

    for i, s in enumerate(symbols, 1):
        try:
            res = compute_asset_metrics(s)
            if res is None:
                continue  # strict: skip incomplete rows
            mrow, srow = res
            assets_rows.append(mrow)
            spark_rows.append(srow)
        except Exception as e:
            logging.info(f"[Assets] {s} skipped: {e}")
        if i % 50 == 0:
            logging.info(f"[Assets] Processed {i}/{len(symbols)}")

    # Spark Data header & write (RAW ok)
    spark_max_len = max((len(r) - 1) for r in spark_rows) if spark_rows else 0
    spark_header = ["symbol"] + [f"c{i}" for i in range(1, spark_max_len + 1)]
    replace_sheet(ws_spark, spark_rows, spark_header, value_input_option="RAW")

    # Assets header
    assets_header = [
        "symbol",
        "last_price",
        "%_down_from_ATH",
        "return_1d_%", "return_7d_%", "return_14d_%",
        "vol_24h",
        "RSI14_15m", "MA60_15m", "MA240_15m",
        "MACD", "MACD_signal", "MACD_hist",
        "sparkline"
    ]

    # Inject SPARKLINE formulas (col N)
    width = max(1, min(SPARK_LEN, spark_max_len))
    spark_formula_tpl = (
        f'=IFERROR(SPARKLINE(OFFSET({SPARK_WS}!$B$1, '
        f'MATCH($A{{ROW}}, {SPARK_WS}!$A$2:$A, 0), 0, 1, {width})), "")'
    )
    for idx in range(len(assets_rows)):
        rownum = idx + 2  # data starts at A2
        assets_rows[idx][-1] = spark_formula_tpl.replace("{ROW}", str(rownum))

    # USER_ENTERED so SPARKLINE formulas are evaluated
    replace_sheet(ws_assets, assets_rows, assets_header, value_input_option="USER_ENTERED")

    # Formatting
    try:
        set_frozen(ws_assets, rows=1)
        format_cell_ranges(ws_assets, [
            ("B2:B", CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="0.0000"))),  # last price
            ("C2:C", CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="0.0000"))),  # % down ATH
            ("D2:F", CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="0.0000"))),  # returns
            ("G2:G", CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="0"))),       # volume
            ("H2:M", CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern="0.0000"))),  # indicators
        ])
        ws_assets.set_basic_filter()
    except Exception:
        pass

    logging.info(f"[Assets] Wrote {len(assets_rows)} complete rows to '{ASSETS_WS}' and spark data to '{SPARK_WS}'.")

# =========================
# Main
# =========================
def main():
    if not _apca_enabled():
        logging.error("APCA_API_KEY_ID / APCA_API_SECRET_KEY are required.")
        sys.exit(1)
    if not GOOGLE_CREDS_JSON:
        logging.error("GOOGLE_CREDS_JSON is required.")
        sys.exit(1)

    key_tail = APCA_API_KEY_ID[-4:] if APCA_API_KEY_ID else "----"
    logging.info(f"[Alpaca] Using API key ending with: {key_tail}")
    logging.info(f"[Alpaca] Data base: {APCA_DATA_BASE} | Feed: {ALPACA_FEED}")
    logging.info(f"[Sheets] Writing to spreadsheet: {GOOGLE_SHEET_NAME}")

    gc = get_client()
    run_assets(gc)
    print("Done: assets metrics + sparkline data")

if __name__ == "__main__":
    main()
