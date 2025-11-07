# ---- build stage (just to cache wheels) ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS deps (certs, basic build tools for any wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip wheel \
 && pip wheel --no-cache-dir -r requirements.txt -w /wheels

# ---- runtime stage ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser
WORKDIR /app

# Install wheels built in the builder stage
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/*

# App code
# Name your script exactly like you saved it (e.g. main.py or app.py)
COPY app.py ./app.py

# Useful defaults (override in your platform/compose)
ENV \
  APCA_API_KEY_ID="" \
  APCA_API_SECRET_KEY="" \
  GOOGLE_CREDS_JSON="" \
  GOOGLE_SHEET_NAME="Trading Log" \
  MAX_SYMBOLS="500" \
  ALPACA_FEED="iex" \
  LOOKBACK_DAYS_1D="450" \
  LOOKBACK_DAYS_15M="60" \
  SPARK_LEN="100" \
  ASSETS_WS="assets" \
  SPARK_WS="spark_data"

# If your platform supports healthchecks, you can add a lightweight one later.

USER appuser

# Run it
CMD ["python", "app.py"]
