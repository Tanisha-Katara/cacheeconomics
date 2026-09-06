FROM python:3.14.7-slim-trixie@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6 AS builder

WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE /src/
COPY harness/cacheeconomics /src/harness/cacheeconomics
COPY collectors /src/collectors
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /src \
    && python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /src/collectors

FROM python:3.14.7-slim-trixie@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

ARG VCS_REF="unknown"
LABEL org.opencontainers.image.title="cacheeconomics collector" \
      org.opencontainers.image.description="Prompt-free network collector kept outside the local analysis package" \
      org.opencontainers.image.source="https://github.com/Tanisha-Katara/cacheeconomics" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels \
      cacheeconomics==0.3.0 cacheeconomics-collector==0.1.0 \
    && addgroup --system collector \
    && adduser --system --ingroup collector collector \
    && mkdir -p /data \
    && chown collector:collector /data \
    && rm -rf /wheels

USER collector
WORKDIR /data
VOLUME ["/data"]

ENTRYPOINT ["cacheeconomics-collector"]
CMD ["--help"]
