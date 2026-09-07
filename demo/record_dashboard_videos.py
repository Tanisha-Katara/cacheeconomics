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
    page.goto(f"{URL}#sign-in", wait_until="domcontentloaded")
    page.locator("#development-token").fill(TOKEN)
    page.locator("#development-token-form button[type=submit]").click()
    page.locator("#workspace").wait_for(state="visible")
    page.locator("#overview-content .kpi-grid").wait_for(state="visible")
    page.evaluate(
        """() => {
          const organization = document.querySelector("#organization-select option:checked");
          const source = document.querySelector("#source-select option:checked");
          if (organization) organization.textContent = "Example workspace";
          if (source) source.textContent = "Example usage source";
          const overviewTitle = document.querySelector("#overview-content .section-intro h2");
          if (overviewTitle) overviewTitle.textContent = "Example usage source";
          document.querySelector("#user-name").textContent = "Demo user";
          document.querySelector("#user-avatar").textContent = "D";
        }"""
    )


def _install_tour_caption(page: Page) -> None:
    page.evaluate(
        """() => {
          const caption = document.createElement("div");
          caption.id = "tour-caption";
          const kicker = document.createElement("span");
          const title = document.createElement("strong");
          caption.append(kicker, title);
          document.body.append(caption);
        }"""
    )


def _caption(page: Page, kicker: str, title: str) -> None:
    page.evaluate(
        """([kicker, title]) => {
          const caption = document.querySelector("#tour-caption");
          caption.classList.remove("is-visible");
          caption.querySelector("span").textContent = kicker;
          caption.querySelector("strong").textContent = title;
          requestAnimationFrame(() => requestAnimationFrame(() => {
            caption.classList.add("is-visible");
          }));
        }""",
        [kicker, title],
    )


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
        reduced_motion="no-preference",
    )
    page = context.new_page()
    recording_started = time.monotonic()
    _open_workspace(page)
    trim_start = max(0.0, time.monotonic() - recording_started)
    _install_tour_caption(page)
    page.wait_for_timeout(400)
    walkthrough(page)
    page.wait_for_timeout(650)
    page.screenshot(path=str(MEDIA / poster), type="jpeg", quality=88)
    video = page.video
    context.close()
    if video is None:
        raise RuntimeError("Playwright did not create a recording")
    subprocess.run(  # noqa: S603 - fixed local ffmpeg command
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{trim_start:.3f}",
            "-i",
            str(video.path()),
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-deadline",
            "good",
            "-cpu-used",
            "2",
            "-row-mt",
            "1",
            "-y",
            str(MEDIA / name),
        ],
        check=True,
    )


def _recommendations(page: Page) -> None:
    _caption(page, "CURRENT USAGE", "See cache reads, writes, and fresh input.")
    page.locator("#overview-content .kpi").nth(3).hover()
    page.wait_for_timeout(1600)
    page.locator('[data-view="recommendations"]').click()
    page.locator("#recommendations-content .recommendation").first.wait_for(
        state="visible"
    )
    _caption(page, "TOP RECOMMENDATION", "See the problem and the change to test.")
    page.wait_for_timeout(2600)
    page.locator("#recommendations-content .recommendation").first.hover()
    page.wait_for_timeout(2200)
    page.locator("#recommendations-content .recommendation").nth(1).hover()
    page.wait_for_timeout(1900)


def _operations(page: Page) -> None:
    page.locator('[data-view="operations"]').click()
    page.locator("#operations-content .volume-chart").wait_for(state="visible")
    _caption(page, "REQUEST HEALTH", "See volume, response time, and errors.")
    page.wait_for_timeout(2500)
    page.locator("#window-select").select_option("24")
    page.locator("#operations-content .volume-chart").wait_for(state="visible")
    page.wait_for_timeout(1900)
    page.locator('[data-view="jobs"]').click()
    page.locator("#jobs-content .job-row").first.wait_for(state="visible")
    _caption(page, "DATA DELIVERY", "Check imports, retries, and source status.")
    page.wait_for_timeout(2600)
    page.locator('[data-view="operations"]').click()
    page.locator("#operations-content .volume-chart").wait_for(state="visible")
    _caption(page, "REQUEST HEALTH", "See volume, response time, and errors.")
    page.wait_for_timeout(900)


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
