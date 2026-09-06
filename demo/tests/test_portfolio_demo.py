from __future__ import annotations

import json

from demo.build_portfolio_packet import (
    DEFAULT_OUTPUT,
    DEMO_ORGANIZATION_ID,
    DEMO_SOURCE_ID,
    DEMO_TOKEN,
    build_packet,
)
from demo.serve_portfolio_demo import _api_response, load_packet


def test_checked_packet_is_rebuilt_by_the_real_synthetic_flow():
    assert build_packet() == json.loads(DEFAULT_OUTPUT.read_text())


def test_packet_is_prompt_free_and_keeps_unreconciled_money_withheld():
    packet = load_packet()
    serialized = json.dumps(packet).lower()
    for forbidden in (
        '"prompt"',
        '"completion"',
        '"messages"',
        '"request_body"',
        '"response_body"',
        '"error_message"',
    ):
        assert forbidden not in serialized
    assert packet["checks"]["stored_prompt_fields"] == 0
    assert packet["checks"]["all_monetary_figures_withheld"] is True

    analysis = packet["responses"]["analysis"]["result"]["analysis"]
    figures = list(analysis["spend"].values())
    figures.append(analysis["total_avoidable_usd_month"])
    for finding in analysis["findings"]:
        figures.extend(
            figure
            for figure in (
                finding["avoidable_usd_window"],
                finding["avoidable_usd_month"],
            )
            if figure is not None
        )
    monetary = [item for item in figures if isinstance(item, dict) and "released" in item]
    assert monetary
    assert all(item["released"] is False for item in monetary)
    assert all(item["amount_usd"] is None for item in monetary)


def test_replay_api_requires_the_demo_identity_and_organization_context():
    packet = load_packet()
    status, _ = _api_response(packet, "GET", "/api/v1/me", {})
    assert status == 401

    auth = {"Authorization": f"Bearer {DEMO_TOKEN}"}
    status, body = _api_response(packet, "GET", "/api/v1/me", auth)
    assert status == 200
    assert body["user"]["display_name"] == "Synthetic Demo User"

    status, _ = _api_response(packet, "GET", "/api/v1/sources", auth)
    assert status == 403
    scoped = auth | {"X-Organization-ID": DEMO_ORGANIZATION_ID}
    status, body = _api_response(packet, "GET", "/api/v1/sources", scoped)
    assert status == 200
    assert body[0]["id"] == DEMO_SOURCE_ID

    lowercase = {
        "authorization": f"Bearer {DEMO_TOKEN}",
        "x-organization-id": DEMO_ORGANIZATION_ID,
    }
    status, _ = _api_response(packet, "GET", "/api/v1/sources", lowercase)
    assert status == 200


def test_replay_api_supports_each_dashboard_window_but_no_mutations():
    packet = load_packet()
    headers = {
        "Authorization": f"Bearer {DEMO_TOKEN}",
        "X-Organization-ID": DEMO_ORGANIZATION_ID,
    }
    for hours in (24, 168, 720):
        path = (
            f"/api/v1/sources/{DEMO_SOURCE_ID}/operations?window_hours={hours}"
        )
        status, body = _api_response(packet, "GET", path, headers)
        assert status == 200
        assert body["window_hours"] == hours

    status, body = _api_response(
        packet,
        "POST",
        f"/api/v1/sources/{DEMO_SOURCE_ID}/jobs",
        headers,
    )
    assert status == 405
    assert body["detail"] == "The portfolio packet is read-only."
