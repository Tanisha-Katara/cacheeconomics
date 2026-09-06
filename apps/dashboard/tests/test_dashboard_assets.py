from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DASHBOARD = ROOT / "apps" / "dashboard"


def _read(name: str) -> str:
    return (DASHBOARD / name).read_text(encoding="utf-8")


def test_dashboard_has_all_phase_three_views_and_states():
    html = _read("index.html")
    javascript = _read("app.js")

    for view in ("overview", "recommendations", "operations", "jobs"):
        assert f'id="view-{view}"' in html
    for state in ("loading", "partial", "behind ingestion", "unavailable", "withheld"):
        assert state in javascript.lower()


def test_tokens_are_not_persisted_or_rendered_as_html():
    javascript = _read("app.js")

    assert "localStorage" not in javascript
    assert "innerHTML" not in javascript
    assert "access_token" in javascript
    assert "sessionStorage.setItem(PKCE_VERIFIER_KEY" in javascript
    assert "sessionStorage.removeItem(PKCE_VERIFIER_KEY" in javascript
    assert 'redirect: "error"' in javascript


def test_dashboard_uses_pkce_and_organization_headers():
    javascript = _read("app.js")

    assert 'code_challenge_method", "S256"' in javascript
    assert 'searchParams.set("audience", state.config.audience)' in javascript
    assert 'headers.set("Authorization"' in javascript
    assert 'headers.set("X-Organization-ID"' in javascript
    assert "amount_usd" not in javascript


def test_reverse_proxy_is_same_origin_and_hardened():
    nginx = _read("default.conf.template")

    assert "location /api/" in nginx
    assert "proxy_pass ${DASHBOARD_API_ORIGIN}/" in nginx
    assert "proxy_set_header Host $proxy_host" in nginx
    assert "object-src 'none'" in nginx
    assert "frame-ancestors 'none'" in nginx
    assert 'X-Frame-Options "DENY"' in nginx
    assert "${DASHBOARD_CONNECT_SRC}" in nginx
    assert "${DASHBOARD_API_ORIGIN}" in nginx


def test_dashboard_container_drops_root():
    dockerfile = _read("Dockerfile")

    assert "USER nginx" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert 'DASHBOARD_API_ORIGIN="http://api:8000"' in dockerfile
