FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential flex bison \
    && rm -rf /var/lib/apt/lists/*

# Клонируем и собираем MatIEC
WORKDIR /src
RUN git clone --depth=1 https://github.com/nucleron/matiec.git . \
    && autoreconf -i 2>/dev/null || true \
    && ./bootstrap 2>/dev/null || true \
    && ./configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install

# ── Runtime image ────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl libfl2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/iec2c /usr/local/bin/iec2c
COPY --from=builder /usr/local/bin/iec2iec /usr/local/bin/iec2iec

# Простой HTTP-сервер для вызова matiec из Python
COPY matiec_server.py /app/matiec_server.py
RUN pip3 install --no-cache-dir --break-system-packages fastapi uvicorn

WORKDIR /app
EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "matiec_server:app", "--host", "0.0.0.0", "--port", "8001"]