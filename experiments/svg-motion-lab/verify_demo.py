"""Verify the replay gallery in a real browser and record a short walkthrough."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


def wait_for_scene(page: Page) -> None:
    """Wait for both isolated SVG documents and the synchronized transport."""
    page.wait_for_function(
        "document.querySelectorAll('.canvas iframe').length === 2 && "
        "!document.querySelector('#pause').disabled && "
        "[...document.querySelectorAll('.canvas iframe')].every(f => "
        "f.contentDocument?.querySelector('svg'))"
    )


def verify(demo: str, output: Path) -> None:
    """Check real controls, every scene, responsive layout and reduced motion."""
    output.mkdir(parents=True, exist_ok=True)
    url = (
        demo
        if demo.startswith(("https://", "http://"))
        else Path(demo).resolve().as_uri()
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1100},
            record_video_dir=str(output),
            record_video_size={"width": 1440, "height": 1100},
        )
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        wait_for_scene(page)
        assert page.locator(".scene").count() == 4
        assert not page.locator("textarea, input[type=text]").count()
        page.locator("#pause").click()
        page.locator("#timeline").fill("500")
        page.locator("#timeline").dispatch_event("input")
        times = page.evaluate(
            "[...document.querySelectorAll('.canvas iframe')].map(f=>f.contentDocument.querySelector('svg').getCurrentTime())"
        )
        assert abs(times[0] - times[1]) < 0.03, times
        assert page.locator("#pause").inner_text() == "Play"
        page.screenshot(path=str(output / "comparison.png"), full_page=True)
        page.locator("#playground").scroll_into_view_if_needed()
        for index in range(4):
            page.locator(".scene").nth(index).click()
            wait_for_scene(page)
            assert page.locator('.scene[aria-pressed="true"]').count() == 1
            page.locator("#replay").click()
            page.wait_for_timeout(4300)
        page.locator("#pause").click()
        assert page.locator("#pause").inner_text() == "Play"
        if errors:
            raise AssertionError(errors)
        video = page.video
        page.close()
        context.close()
        video.save_as(output / "comparison.webm")
        mobile = browser.new_context(
            viewport={"width": 390, "height": 844}, reduced_motion="reduce"
        )
        page = mobile.new_page()
        page.goto(url)
        wait_for_scene(page)
        assert page.locator("#pause").inner_text() == "Play"
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator(".scene").nth(2).focus()
        page.keyboard.press("Enter")
        wait_for_scene(page)
        assert page.locator(".scene").nth(2).get_attribute("aria-pressed") == "true"
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        mobile.close()
        browser.close()
    print(
        "Verified: four scenes, paired rendering, synchronized scrub, replay, pause, keyboard, mobile and reduced motion"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", default="demo/index.html")
    parser.add_argument("--output", type=Path, default=Path("output/playwright"))
    args = parser.parse_args()
    verify(args.demo, args.output)
