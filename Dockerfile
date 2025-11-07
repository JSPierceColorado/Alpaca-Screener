# ---- build stage (cache wheels) ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only reqs first to maximize layer cache
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir pip wheel \
 && pip wheel --no-cache-dir -r requirements.txt -w /wheels

# ---- runtime stage ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser
WORKDIR /app

# Install deps from prebuilt wheels
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/*

# App code (just main.py)
COPY main.py ./main.py

# ⚠️ Do NOT bake secrets into the image (no ENV for keys here).
# Pass APCA_API_KEY_ID, APCA_API_SECRET_KEY, GOOGLE_CREDS_JSON, GOOGLE_SHEET_NAME, MAX_SYMBOLS at runtime.

USER appuser

CMD ["python", "main.py"]
