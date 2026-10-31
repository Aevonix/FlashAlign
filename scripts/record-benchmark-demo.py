# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Capture an exported benign playback to WebM using optional Playwright."""
import argparse
from pathlib import Path
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("output", type=Path, help="New output .webm path")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.suffix != ".webm":
        parser.error("Choose a new .webm output file")
    if not (320 <= args.width <= 3840 and 240 <= args.height <= 3840):
        parser.error("Capture dimensions must fit within 320–3840 by 240–3840 pixels")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        size = {"width": args.width, "height": args.height}
        context = browser.new_context(viewport=size,
                                      record_video_dir=str(args.output.parent),
                                      record_video_size=size)
        page = context.new_page()
        page.goto(args.html.resolve().as_uri())
        page.wait_for_function("!!window.flashalignPlayback")
        duration = page.evaluate("window.flashalignPlayback.duration")
        if not isinstance(duration, (float, int)) or duration < 0 or duration > 1800:
            raise ValueError("Playback duration must be between zero and 30 minutes")
        page.evaluate("window.flashalignPlayback.play()")
        # Short polling keeps the process interruptible during a long response.
        deadline = time.monotonic() + duration + 2
        while time.monotonic() < deadline:
            page.wait_for_timeout(min(1000, max(1, (deadline - time.monotonic()) * 1000)))
        video = page.video
        context.close()
        video.save_as(str(args.output))
        original = Path(video.path())
        if original.resolve() != args.output.resolve():
            original.unlink(missing_ok=True)
        args.output.chmod(0o600)
        browser.close()
    print("Saved real-time playback capture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
