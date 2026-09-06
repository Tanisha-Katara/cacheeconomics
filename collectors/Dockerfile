FROM python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS builder

WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE /src/
COPY harness/cacheeconomics /src/harness/cacheeconomics
COPY collectors /src/collectors
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /src \
    && python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /src/collectors

FROM python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

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
