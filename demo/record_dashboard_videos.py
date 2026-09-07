#!/usr/bin/env python3
"""Record the checked synthetic dashboard packet as two short product films."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Browser, Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "apps/dashboard/media"
URL = "http://127.0.0.1:8765/"
TOKEN = "synthetic-demo-only"
VIEWPORT = {"width": 1280, "height": 720}


def _browser_path() -> str | None:
    configured = os.environ.get("CACHEECONOMICS_BROWSER")
    if configured:
        return configured
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac_chrome.is_file():
        return str(mac_chrome)
    return shutil.which("google-chrome") or shutil.which("chromium")


def _wait_for_demo() -> None:
    for _ in range(80):
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed loopback demo URL
                f"{URL}api/v1/dashboard/config", timeout=0.25
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("the synthetic demo server did not start")


def _open_workspace(page: Page) -> None:
    page.goto(f"{URL}#sign-in", wait_until="networkidle")
    page.locator("#development-token").fill(TOKEN)
    page.locator("#development-token-form button[type=submit]").click()
    page.locator("#workspace").wait_for(state="visible")
    page.locator("#overview-content .kpi-grid").wait_for(state="visible")


def _record(
    browser: Browser,
    temporary: Path,
    name: str,
    poster: str,
    walkthrough,
) -> None:
    context = browser.new_context(
        viewport=VIEWPORT,
        record_video_dir=str(temporary),
        record_video_size=VIEWPORT,
        reduced_motion="reduce",
    )
    page = context.new_page()
    _open_workspace(page)
    page.wait_for_timeout(900)
    walkthrough(page)
    page.wait_for_timeout(900)
    page.screenshot(path=str(MEDIA / poster), type="jpeg", quality=88)
    video = page.video
    context.close()
    if video is None:
        raise RuntimeError("Playwright did not create a recording")
    shutil.copyfile(video.path(), MEDIA / name)


def _recommendations(page: Page) -> None:
    page.locator("#overview-content .kpi").nth(3).hover()
    page.wait_for_timeout(1900)
    page.locator('[data-view="recommendations"]').click()
    page.locator("#recommendations-content .recommendation").first.wait_for(
        state="visible"
    )
    page.wait_for_timeout(3300)
    page.locator("#recommendations-content .recommendation").first.hover()
    page.wait_for_timeout(3100)
    page.locator("#recommendations-content .recommendation").nth(1).hover()
    page.wait_for_timeout(2700)


def _operations(page: Page) -> None:
    page.locator('[data-view="operations"]').click()
    page.locator("#operations-content .volume-chart").wait_for(state="visible")
    page.wait_for_timeout(3500)
    page.locator("#window-select").select_option("24")
    page.locator("#operations-content .volume-chart").wait_for(state="visible")
    page.wait_for_timeout(2900)
    page.locator('[data-view="jobs"]').click()
    page.locator("#jobs-content .job-row").first.wait_for(state="visible")
    page.wait_for_timeout(3900)
    page.locator('[data-view="operations"]').click()
    page.locator("#operations-content .volume-chart").wait_for(state="visible")
    page.wait_for_timeout(1200)


def main() -> int:
    MEDIA.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(  # noqa: S603 - fixed repository script
        [sys.executable, str(ROOT / "demo/serve_portfolio_demo.py")],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_demo()
        with tempfile.TemporaryDirectory(prefix="cacheeconomics-films-") as directory:
            with sync_playwright() as playwright:
                executable = _browser_path()
                browser = playwright.chromium.launch(
                    headless=True,
                    executable_path=executable,
                )
                try:
                    temporary = Path(directory)
                    _record(
                        browser,
                        temporary,
                        "recommendations-tour.webm",
                        "recommendations-tour-poster.jpg",
                        _recommendations,
                    )
                    _record(
                        browser,
                        temporary,
                        "operations-tour.webm",
                        "operations-tour-poster.jpg",
                        _operations,
                    )
                finally:
                    browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
