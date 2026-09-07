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


def test_public_site_explains_the_product_before_sign_in():
    html = _read("index.html")

    assert html.index('id="landing-view"') < html.index('id="auth-view"')
    assert 'id="roi-calculator"' in html
    assert 'id="demo"' in html
    assert 'id="security"' in html
    assert 'id="faq"' in html
    assert "Spend less on repeated context." in html
    assert "Prompt bodies rejected" in html
    assert "checked synthetic data" in html


def test_roi_model_is_an_adjustable_illustrative_scenario():
    html = _read("index.html")
    javascript = _read("app.js")

    for field in (
        "roi-baseline",
        "roi-repeat",
        "roi-discount",
    ):
        assert f'id="{field}"' in html
    assert "ILLUSTRATIVE STARTING POINT" in html
    assert 'id="roi-baseline"' in html and 'value="25000"' in html
    assert "baseline * repeatShare * discount" in javascript
    assert "cache-write premiums" in html


def test_landing_motion_is_visible_and_respects_reduced_motion():
    javascript = _read("app.js")
    stylesheet = _read("styles.css")

    assert 'classList.add("landing-ready")' in javascript
    assert "requestAnimationFrame" in javascript
    assert "preview-scan" in stylesheet
    assert "hero-line-in" in stylesheet
    assert "prefers-reduced-motion: reduce" in stylesheet


def test_product_films_are_checked_in_and_clearly_synthetic():
    html = _read("index.html")
    media = DASHBOARD / "media"

    for name in (
        "recommendations-tour.webm",
        "recommendations-tour-poster.jpg",
        "operations-tour.webm",
        "operations-tour-poster.jpg",
        "social-preview.png",
    ):
        assert (media / name).stat().st_size > 1_000
        assert f"/media/{name}" in html
    assert html.count("SYNTHETIC PRODUCT WALKTHROUGH") == 1
    assert 'property="og:image:width" content="1200"' in html
    assert 'property="og:image:height" content="630"' in html
    assert (DASHBOARD / "favicon.svg").is_file()
    assert 'href="/favicon.svg"' in html


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
    assert "COPY apps/dashboard/media /usr/share/nginx/html/media" in dockerfile
    assert "COPY apps/dashboard/favicon.svg /usr/share/nginx/html/favicon.svg" in dockerfile


def test_withheld_money_stays_compact_in_the_interface():
    javascript = _read("app.js")

    assert 'if (!figure.released || figure.release_state === "withheld") return "Withheld";' in javascript
    assert "spend?.withheld_because" in javascript
