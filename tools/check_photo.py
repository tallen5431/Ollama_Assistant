#!/usr/bin/env python3
"""Show what the app can read out of a photo, and what it cannot.

"Photo details is on but no location came through" has two very different
causes: the file never had a position, or the app failed to read one. From
inside the browser they look identical. This points the *shipped* parser — the
same JavaScript the page runs, not a second implementation that might disagree
— at a file on disk and prints everything it found.

    .venv/bin/python tools/check_photo.py ~/Downloads/odometer.jpg
    .venv/bin/python tools/check_photo.py *.jpg

Needs node, for the same reason the tests do: running the real parser is the
only way to be sure the answer matches what the app would do.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat_ui  # noqa: E402
import web  # noqa: E402

OK, BAD, WARN = "  ✅", "  ❌", "  ⚠️ "

# What the browser reads. Matching it matters: a file whose EXIF sits past this
# would be reported here exactly as the app sees it — which is the point.
HEAD_BYTES = 256 * 1024


def parser_source() -> str:
    """The EXIF functions exactly as the page ships them."""
    page = re.search(r"<script>(.*?)</script>", chat_ui.render_page("t"), re.S).group(1)
    start = page.index("      const EXIF_HEAD_BYTES")
    end = page.index("      async function toAttachment", start)
    # readExif itself needs a File and a window; everything below it is pure.
    return re.sub(r"function readExif[\s\S]*?\n      }\n", "", page[start:end])


def read(path: str) -> object:
    """Run the shipped parser over one file. None when it found nothing."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as script:
        script.write(
            parser_source()
            + "\nconst fs = require('fs');"
            + f"\nconst b = fs.readFileSync({json.dumps(path)}).slice(0, {HEAD_BYTES});"
            + "\nconst v = new DataView(b.buffer, b.byteOffset, b.byteLength);"
            + "\ntry { console.log(JSON.stringify(parseExif(v))); }"
            + "\ncatch (e) { console.log(JSON.stringify({__error: String(e)})); }\n"
        )
        name = script.name
    try:
        out = subprocess.run(["node", name], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(name)
    if out.returncode != 0:
        return {"__error": out.stderr.strip()[:400]}
    return json.loads(out.stdout.strip() or "null")


# Printed in this order, with the name a person would use rather than the EXIF one.
FIELDS = (
    ("taken", "Taken"),
    ("offset", "Time zone"),
    ("utc", "GPS clock (UTC)"),
    ("lat", "Latitude"),
    ("lon", "Longitude"),
    ("altitude", "Altitude (m)"),
    ("accuracy", "Position error (m)"),
    ("speed", "Speed"),
    ("heading", "Facing"),
    ("camera", "Camera"),
    ("lens", "Lens"),
    ("width", "Width"),
    ("height", "Height"),
    ("exposure", "Exposure"),
    ("aperture", "Aperture"),
    ("focal", "Focal length"),
    ("focal35", "35 mm equivalent"),
    ("iso", "ISO"),
    ("flash", "Flash"),
    ("orientation", "Orientation"),
    ("software", "Written by"),
    ("artist", "Artist"),
    ("copyright", "Copyright"),
)


def report(path: str) -> bool:
    """Print one file's metadata. Returns whether a position was found."""
    print(f"\n{path}")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        print(f"{BAD} cannot read it: {exc}")
        return False
    print(f"     {size / 1024:.0f} KB")

    found = read(path)
    if isinstance(found, dict) and found.get("__error"):
        print(f"{BAD} the parser failed: {found['__error']}")
        return False
    if not found:
        print(f"{BAD} no EXIF at all.")
        print("     Screenshots have none. Neither does anything re-encoded on the")
        print("     way here — a messaging app, or Google Photos' 'share' rather")
        print("     than 'download'. Send the original file off the phone.")
        return False

    for key, label in FIELDS:
        if key in found:
            print(f"{OK} {label + ':':<20}{found[key]}")

    if "lat" in found and "lon" in found:
        if found["lat"] == 0 and found["lon"] == 0:
            print(f"{WARN} the position is exactly 0, 0 — that is what a camera writes")
            print("     for a GPS block it never got a fix for. Treated as no position.")
            return False
        print(f"{OK} location is present, and the app will send it.")
        print(f"     https://www.openstreetmap.org/?mlat={found['lat']}&mlon={found['lon']}#map=16")
        return True

    print(f"{WARN} no location in this file.")
    if "taken" in found:
        print("     The rest of the EXIF read fine, so this is the file, not the app:")
        print("     the camera did not record a position. On Android that is Camera →")
        print("     Settings → Location, and the app also needs the location permission.")
        print("     On iPhone it is Settings → Privacy → Location Services → Camera.")
    return False


def main() -> int:
    if shutil.which("node") is None:
        print(f"{BAD} node is not installed, and this runs the app's own parser under it.")
        return 2
    paths = sys.argv[1:]
    if not paths:
        print(__doc__.strip())
        return 2

    located = 0
    for path in paths:
        if report(path):
            located += 1

    print(f"\n{len(paths)} file(s), {located} with a position.")
    if located:
        # The same rendering the model is handed, so what is on screen here is
        # exactly what it will be told.
        entries = [read(p) for p in paths]
        entries = [e if isinstance(e, dict) and not e.get("__error") else None for e in entries]
        rendered = web.image_metadata(entries)
        if rendered:
            print("\nWhat the model would be told:\n")
            for line in rendered.splitlines()[2:]:
                print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
