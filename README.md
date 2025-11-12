# Alpaca Screener → Google Sheets

A lightweight Python 3 script that pulls **tradable US equities** from **Alpaca**, computes swing‑trading metrics (RSI, MA60/MA240, MACD, ATH drawdown, short‑horizon returns), and writes a clean, filterable dashboard into **Google Sheets** — complete with **sparklines**.

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-stable-brightgreen" />
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-blue" />
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

---

## ✨ Features

* **End‑to‑end pipeline**: fetch assets → compute metrics → write two worksheets (`Alpaca-Screener`, `spark_data`).
* **Strict completeness**: skips any symbol that can’t compute all metrics (ensures a clean sheet).
* **Retry + backoff**: resilient HTTP layer with pacing and `429` handling.
* **Indicator set**: RSI(14) on 15‑minute closes, MA60/MA240, MACD(12,26,9), ATH drawdown, 1/7/14‑day returns, last daily volume.
* **Sparklines**: Google Sheets `SPARKLINE` formulas auto‑injected for 15‑minute close trends.
* **Formatting**: number formats + header freeze + basic filter applied automatically.

---

## 🧭 Sheet Layout

The script writes to your spreadsheet named via `GOOGLE_SHEET_NAME` (default: **Trading Log**) and creates/updates two tabs:

* **`Alpaca-Screener`** — high‑level dashboard

  * `symbol`, `last_price`, `%_down_from_ATH`, `return_1d_%`, `return_7d_%`, `return_14d_%`,
    `vol_24h`, `RSI14_15m`, `MA60_15m`, `MA240_15m`, `MACD`, `MACD_signal`, `MACD_hist`, `sparkline`
* **`spark_data`** — backing data for sparklines

  * `symbol`, followed by up to `SPARK_LEN` recent 15‑minute closes

The sparkline formula placed in column **N** of `Alpaca-Screener` looks like:

```gs
=IFERROR(SPARKLINE(OFFSET(spark_data!$B$1, MATCH($A2, spark_data!$A$2:$A, 0), 0, 1, <width>)), "")
```

Where `<width>` is the min of `SPARK_LEN` and the available closes.

---

## 📦 Requirements

* Python **3.11+**
* Google Service Account with Drive/Sheets access
* Alpaca **API keys** with access to Trading API and Data API v2 (IEX or SIP feed)

Install deps:

```bash
pip install -r requirements.txt
```

Minimal requirements (if you manage your own):

```
requests
gspread
google-auth
gspread-formatting
```

---

## 🔧 Configuration (Environment Variables)

| Variable              | Default                       | Description                                                              |
| --------------------- | ----------------------------- | ------------------------------------------------------------------------ |
| `APCA_API_KEY_ID`     | —                             | Alpaca API key ID. **Required**.                                         |
| `APCA_API_SECRET_KEY` | —                             | Alpaca API secret. **Required**.                                         |
| `GOOGLE_CREDS_JSON`   | —                             | JSON **string** of the Google service account credentials. **Required**. |
| `GOOGLE_SHEET_NAME`   | `Trading Log`                 | Target spreadsheet name.                                                 |
| `MAX_SYMBOLS`         | `1000`                        | Cap on symbols processed.                                                |
| `HTTP_TIMEOUT`        | `15`                          | Per‑request timeout (sec).                                               |
| `APCA_API_BASE`       | `https://api.alpaca.markets`  | Trading API base.                                                        |
| `APCA_DATA_BASE`      | `https://data.alpaca.markets` | Data API v2 base.                                                        |
| `ALPACA_FEED`         | `iex`                         | `iex` or `sip` for market data source.                                   |
| `LOOKBACK_DAYS_1D`    | `450`                         | Daily bar lookback for ATH/returns/volume.                               |
| `LOOKBACK_DAYS_15M`   | `60`                          | 15‑minute bar lookback for indicators.                                   |
| `SPARK_LEN`           | `100`                         | Number of 15‑minute closes in sparkline.                                 |
| `ASSETS_WS`           | `Alpaca-Screener`             | Dashboard worksheet title.                                               |
| `SPARK_WS`            | `spark_data`                  | Sparkline data worksheet title.                                          |

### Example `.env`

```env
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
GOOGLE_CREDS_JSON='{"type":"service_account", ... }'
GOOGLE_SHEET_NAME=Trading Log
MAX_SYMBOLS=800
```

> Tip: If your platform only supports **env vars**, keep `GOOGLE_CREDS_JSON` as a single minified line.

---

## 🔐 Google Setup (Service Account)

1. Create a **Service Account** in Google Cloud and generate a JSON key.
2. Share your target Google Sheet with the service account’s email (Editor). You can create the sheet manually or let the script create the worksheets in an existing spreadsheet.
3. Put the JSON contents into `GOOGLE_CREDS_JSON` (as a single string).

Required scopes used by this script:

* `https://www.googleapis.com/auth/spreadsheets`
* `https://www.googleapis.com/auth/drive`

---

## ▶️ Running

### Local

```bash
export $(grep -v '^#' .env | xargs)   # optional convenience
python main.py
```

### Docker (example)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

```bash
docker build -t alpaca-screener .
docker run --rm \
  -e APCA_API_KEY_ID \
  -e APCA_API_SECRET_KEY \
  -e GOOGLE_CREDS_JSON \
  -e GOOGLE_SHEET_NAME \
  alpaca-screener
```

### Railway / Cloud Runners

* Add all env vars in the dashboard.
* Use Python 3.12 or newer base.
* Provide persistent logs if you want to track progress (`INFO` level).

---

## 🧮 Indicators & Logic

* **RSI(14)** on 15‑minute closes
* **MA60 / MA240** on 15‑minute closes
* **MACD(12,26,9)** on 15‑minute closes (returns `MACD`, `signal`, `hist`)
* **ATH drawdown**: `%_down_from_ATH = (ATH − last_close) / ATH`
* **Short‑horizon returns**: 1d, 7d, 14d vs the close `n` days ago
* **Volume**: uses the last **daily** volume as `vol_24h`

> The pipeline is **strict**: if *any* metric is missing or non‑finite for a symbol, that row is skipped to keep your dashboard clean.

---

## 🧱 Architecture (at a glance)

```
┌──────────────┐        ┌──────────────────┐        ┌────────────────────┐
│ Alpaca APIs  │  bars  │  compute metrics │  rows  │ Google Sheets      │
│  - trading   ├────────►  RSI/MA/MACD/    ├────────►  Alpaca-Screener   │
│  - data v2   │ assets │  ATH/returns     │        │  spark_data        │
└──────────────┘        └──────────────────┘        └────────────────────┘
        ▲                         │                              ▲
        │ retries/pacing          │ spark series                 │ formulas
        └─────────────────────────┴──────────────────────────────┘
```

---

## ⚙️ Internals & Notes

* **Retries & pacing**: automatic backoff on 429/5xx, respect `Retry-After`, light per‑host pacing.
* **Sizing**: worksheets auto‑grow before writes; only columns **A:N** are cleared/rewritten on the dashboard.
* **Formatting**: price/percent/indicator number formats applied; header row frozen; basic filter enabled.
* **Feed selection**: set `ALPACA_FEED=iex` (default) or `sip` based on your data plan.

---

## ⏱ Scheduling

Run periodically via cron or your platform’s scheduler, e.g. hourly on weekdays:

```cron
# Every weekday at 20 and 50 past the hour
20,50 * * * 1-5 /usr/bin/python /app/main.py >> /var/log/assets.log 2>&1
```

---

## 🧰 Troubleshooting

* **"API keys missing"**: ensure `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` are set.
* **"GOOGLE_CREDS_JSON is required"**: pass the entire JSON (minified) as an env var.
* **Service account can’t write**: share the Spreadsheet with the service account email.
* **Empty results**: your Alpaca plan/feed may restrict data; verify `ALPACA_FEED` and market hours.
* **429 rate limits**: script already backs off; consider lowering `MAX_SYMBOLS` or increasing schedule spacing.

---

## 🔒 Security

* Treat `GOOGLE_CREDS_JSON` and Alpaca secrets as **sensitive**. Prefer platform‑level secrets managers.
* Avoid committing credentials. Use environment variables or secret stores.

---

## 🤝 Contributing

PRs and issues welcome! Please include reproduction steps and environment details. For larger changes, consider opening a discussion first.

---

## 📜 License

MIT — see `LICENSE`.

---

## 🙌 Acknowledgements

* [Alpaca Markets](https://alpaca.markets/) for trading & market data APIs.
* [gspread](https://github.com/burnash/gspread) and `gspread-formatting` for Sheets ergonomics.
