# Use slim Python base
FROM python:3.12-slim

# Prevent buffered stdout/stderr (helpful on Railway)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first for better layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy project files
COPY main.py ./

# -----------------------------
# Environment variable defaults
# -----------------------------

# Alpaca (override with your real keys in Railway)
ENV APCA_API_KEY_ID="" \
    APCA_API_SECRET_KEY="" \
    APCA_API_BASE_URL="https://paper-api.alpaca.markets" \
    APCA_DATA_BASE_URL="https://data.alpaca.markets"

# Google Sheets
# NOTE: Set GOOGLE_CREDS_JSON in Railway as a variable containing your full service account JSON string
ENV GOOGLE_SHEET_NAME="Trading Log" \
    GOOGLE_CREDS_JSON=""

# Kraken (optional)
ENV KRAKEN_API_KEY="" \
    KRAKEN_API_SECRET=""

# Runtime tuning
ENV REFRESH_MINUTES=30 \
    BATCH_SIZE=50 \
    SLEEP_BETWEEN_BATCHES=5

# Default command
CMD ["python", "main.py"]
