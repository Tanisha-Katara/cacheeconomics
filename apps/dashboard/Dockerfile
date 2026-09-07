FROM nginx:1.30.4-alpine3.24@sha256:dc5069ad14f19660b141b21236140b91656bf89bbc3e2417c70ae650cd66104c

ARG VCS_REF="unknown"
LABEL org.opencontainers.image.title="cacheeconomics dashboard" \
      org.opencontainers.image.description="Static multi-tenant cacheeconomics dashboard" \
      org.opencontainers.image.source="https://github.com/Tanisha-Katara/cacheeconomics" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DASHBOARD_CONNECT_SRC="'self'" \
    DASHBOARD_API_ORIGIN="http://api:8000" \
    NGINX_ENVSUBST_FILTER="DASHBOARD_(CONNECT_SRC|API_ORIGIN)"

# The pinned upstream image currently carries libuuid 2.42.1-r0. Pin the
# Alpine security update as well so rebuilding this image cannot regress to
# the seven fixed util-linux vulnerabilities detected by the image scan.
RUN apk add --no-cache --upgrade 'libuuid=2.42.3-r1'

COPY apps/dashboard/nginx.conf /etc/nginx/nginx.conf
COPY apps/dashboard/default.conf.template /etc/nginx/templates/default.conf.template
COPY apps/dashboard/index.html /usr/share/nginx/html/index.html
COPY apps/dashboard/styles.css /usr/share/nginx/html/styles.css
COPY apps/dashboard/app.js /usr/share/nginx/html/app.js
COPY apps/dashboard/favicon.svg /usr/share/nginx/html/favicon.svg
COPY apps/dashboard/media /usr/share/nginx/html/media

RUN rm -f /etc/nginx/conf.d/default.conf \
    && mkdir -p /tmp/client_temp /tmp/proxy_temp /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp \
    && chown -R nginx:nginx /etc/nginx/conf.d /usr/share/nginx/html /tmp/client_temp /tmp/proxy_temp /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp

USER nginx

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1:8080/ || exit 1
