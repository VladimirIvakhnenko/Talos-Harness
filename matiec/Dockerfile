FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git \
    build-essential autoconf automake libtool flex bison \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 https://github.com/sm1820/matiec.git /src
WORKDIR /src
RUN autoreconf -i \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install \
    && if [ ! -d /usr/local/lib/matiec ]; then \
         mkdir -p /usr/local/lib/matiec; \
         if [ -d lib ]; then cp -r lib/. /usr/local/lib/matiec/; fi; \
       fi

# ── Runtime image ────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl libfl2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/iec2c    /usr/local/bin/iec2c
COPY --from=builder /usr/local/bin/iec2iec  /usr/local/bin/iec2iec
COPY --from=builder /usr/local/lib/matiec   /usr/local/lib/matiec

COPY matiec_server.py /app/matiec_server.py
RUN pip3 install --no-cache-dir --break-system-packages fastapi uvicorn

WORKDIR /app
EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "matiec_server:app", "--host", "0.0.0.0", "--port", "8001"]
