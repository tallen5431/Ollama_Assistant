"""Reading a photo's EXIF, from the JPEG bytes to the model's context.

The browser re-encodes every upload through a canvas, which produces clean
pixels and no metadata whatsoever. So "when and where was this taken?" can only
be answered if the facts are read in the page, before that re-encode, and sent
alongside — which is what these cover, end to end:

  * the shipped JavaScript parser, run under node against real EXIF JPEGs built
    here in both byte orders,
  * the prose it turns into,
  * that it reaches the answering model,
  * and that the coordinates do not reach the search planner, which is the one
    path in this app that talks to the internet.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess

import pytest

import app as app_module
import chat_ui
import store
import web


# --------------------------------------------------------------------------
# A real EXIF JPEG, built here so the parser is tested against bytes rather
# than against a fixture that agrees with it by construction.
# --------------------------------------------------------------------------

def build_jpeg(little_endian: bool = False, gps: bool = True, rich: bool = False,
               fill_bytes: bool = False, jfif: bool = False, thumbnail: bool = False,
               zero_gps: bool = False) -> bytes:
    """A JPEG whose APP1 segment carries camera, timestamp and position.

    ``rich`` adds the rest of what a phone writes — the time-zone offset, the
    GPS clock, the lens, the exposure settings. ``fill_bytes``, ``jfif`` and
    ``thumbnail`` reproduce the surrounding shape of a real file rather than the
    minimal one: a JFIF segment first, legal 0xFF padding before a marker, and a
    thumbnail IFD hanging off IFD0's next-IFD pointer.
    """
    e = "<" if little_endian else ">"
    order = b"II" if little_endian else b"MM"

    def rational(n: int, d: int = 1) -> bytes:
        return struct.pack(e + "II", n, d)

    def ifd_builder():
        rows, blob = [], bytearray()

        def add(tag, typ, count, payload=None, inline=None):
            nonlocal blob
            if inline is not None:
                rows.append((tag, typ, count, inline, True))
            else:
                rows.append((tag, typ, count, len(blob), False))
                blob += payload
        return rows, blob, add

    gps_rows, gps_blob, gadd = ifd_builder()
    if gps:
        degrees = (rational(0), rational(0), rational(0)) if zero_gps else None
        gadd(1, 2, 2, inline=b"N\x00\x00\x00")                       # latitude ref
        gadd(2, 5, 3, payload=b"".join(degrees) if degrees else
             rational(51) + rational(30) + rational(3600, 100))
        gadd(3, 2, 2, inline=b"W\x00\x00\x00")                       # longitude ref
        gadd(4, 5, 3, payload=b"".join(degrees) if degrees else
             rational(0) + rational(7) + rational(3900, 100))
        gadd(5, 1, 1, inline=b"\x00\x00\x00\x00")                    # above sea level
        gadd(6, 5, 1, payload=rational(4200, 100))                   # altitude
        if rich:
            gadd(7, 5, 3, payload=rational(17) + rational(42) + rational(700, 100))
            gadd(12, 2, 2, inline=b"K\x00\x00\x00")                  # speed in km/h
            gadd(13, 5, 1, payload=rational(1234, 100))              # 12.34 km/h
            gadd(16, 2, 2, inline=b"T\x00\x00\x00")                  # true north
            gadd(17, 5, 1, payload=rational(27350, 100))             # facing 273.5°
            gadd(29, 2, 11, payload=b"2026:07:14\x00")               # GPSDateStamp
            gadd(31, 5, 1, payload=rational(800, 100))               # 8 m of error

    exif_rows, exif_blob, eadd = ifd_builder()
    eadd(0x9003, 2, 20, payload=b"2026:07:14 18:42:07\x00")          # DateTimeOriginal
    if rich:
        eadd(0x9010, 2, 7, payload=b"+01:00\x00")                    # OffsetTime
        eadd(0x829A, 5, 1, payload=rational(1, 120))                 # 1/120 s
        eadd(0x829D, 5, 1, payload=rational(180, 100))               # f/1.8
        eadd(0x8827, 3, 1, inline=struct.pack(e + "HH", 64, 0))      # ISO 64
        eadd(0x920A, 5, 1, payload=rational(669, 100))               # 6.69 mm
        eadd(0xA405, 3, 1, inline=struct.pack(e + "HH", 24, 0))      # 24 mm equivalent
        eadd(0x9209, 3, 1, inline=struct.pack(e + "HH", 1, 0))       # flash fired
        eadd(0xA002, 4, 1, inline=struct.pack(e + "I", 4080))        # width
        eadd(0xA003, 4, 1, inline=struct.pack(e + "I", 3072))        # height
        eadd(0xA434, 2, 20, payload=b"Pixel 8 back camera\x00")      # LensModel

    ifd0_rows, ifd0_blob, iadd = ifd_builder()
    iadd(0x010F, 2, 7, payload=b"Google\x00")                        # Make
    iadd(0x0110, 2, 9, payload=b"Pixel 8\x00\x00")                   # Model
    if rich:
        iadd(0x0112, 3, 1, inline=struct.pack(e + "HH", 6, 0))       # rotated 90°
        iadd(0x0131, 2, 9, payload=b"HDR+ 1.0\x00")                  # Software
        iadd(0x013B, 2, 3, inline=b"tj\x00\x00")                     # Artist

    # Lay the IFDs out end to end. Every pointer is relative to the TIFF header,
    # so the sizes have to be settled before any of them can be written.
    extra_count = 2 if gps else 1
    ifd0_at = 8
    ifd0_size = 2 + (len(ifd0_rows) + extra_count) * 12 + 4
    ifd0_blob_at = ifd0_at + ifd0_size
    exif_at = ifd0_blob_at + len(ifd0_blob)
    exif_size = 2 + len(exif_rows) * 12 + 4
    exif_blob_at = exif_at + exif_size
    gps_at = exif_blob_at + len(exif_blob)
    gps_size = 2 + len(gps_rows) * 12 + 4
    gps_blob_at = gps_at + gps_size
    ifd1_at = gps_blob_at + len(gps_blob)

    def pack(rows, blob_at, extra=(), next_ifd=0):
        all_rows = list(rows) + list(extra)
        out = struct.pack(e + "H", len(all_rows))
        for tag, typ, count, value, inline in all_rows:
            out += struct.pack(e + "HHI", tag, typ, count)
            out += value if inline else struct.pack(e + "I", blob_at + value)
        return out + struct.pack(e + "I", next_ifd)

    extras = [(0x8769, 4, 1, struct.pack(e + "I", exif_at), True)]
    if gps:
        extras.append((0x8825, 4, 1, struct.pack(e + "I", gps_at), True))

    tiff = order + struct.pack(e + "HI", 42, ifd0_at)
    tiff += pack(ifd0_rows, ifd0_blob_at, extras, ifd1_at if thumbnail else 0)
    tiff += bytes(ifd0_blob)
    tiff += pack(exif_rows, exif_blob_at) + bytes(exif_blob)
    tiff += pack(gps_rows, gps_blob_at) + bytes(gps_blob)
    if thumbnail:
        tiff += pack([(0x0103, 3, 1, struct.pack(e + "HH", 6, 0), True)], 0)

    app1 = b"Exif\x00\x00" + tiff
    head = b""
    if jfif:
        payload = b"JFIF\x00\x01\x02\x00\x00\x01\x00\x01\x00\x00"
        head += b"\xff\xe0" + struct.pack(">H", len(payload) + 2) + payload
    if fill_bytes:
        head += b"\xff\xff"        # legal padding before the next marker
    segment = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    # SOI, the segments, then a start-of-scan the parser must stop at.
    return b"\xff\xd8" + head + segment + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xff\xd9"


# --------------------------------------------------------------------------
# The shipped parser, under node
# --------------------------------------------------------------------------

def _parser_source() -> str:
    """The EXIF functions exactly as the browser receives them."""
    page = re.search(r"<script>(.*?)</script>", chat_ui.render_page("t"), re.S).group(1)
    at = page.index("function parseExif")
    return page[at:page.index("// Degrees/minutes/seconds", at)] + _dms_of(page)


def _dms_of(page: str) -> str:
    at = page.index("// Degrees/minutes/seconds")
    return page[at:page.index("\n      }", at) + len("\n      }")]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestTheShippedParser:
    """Run the real functions over real bytes, not a Python reimplementation."""

    def _parse(self, tmp_path, data: bytes):
        blob = tmp_path / "in.bin"
        blob.write_bytes(data)
        script = tmp_path / "run.js"
        script.write_text(
            _parser_source()
            + "\nconst fs = require('fs');"
            + f"\nconst b = fs.readFileSync({json.dumps(str(blob))});"
            + "\nconst v = new DataView(b.buffer, b.byteOffset, b.byteLength);"
            + "\nconsole.log(JSON.stringify(parseExif(v)));\n"
        )
        out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout.strip())

    @pytest.mark.parametrize("little_endian", [False, True])
    def test_camera_time_and_place_come_back(self, tmp_path, little_endian):
        got = self._parse(tmp_path, build_jpeg(little_endian=little_endian))
        assert got["camera"] == "Google Pixel 8"
        assert got["taken"] == "2026:07:14 18:42:07"
        # 51° 30' 36" N, 0° 7' 39" W — Westminster.
        assert got["lat"] == pytest.approx(51.51, abs=1e-6)
        assert got["lon"] == pytest.approx(-0.1275, abs=1e-6)
        assert got["altitude"] == 42

    def test_west_and_south_come_back_negative(self, tmp_path):
        got = self._parse(tmp_path, build_jpeg())
        assert got["lon"] < 0, "a W reference must flip the sign"

    def test_a_photo_without_gps_still_reports_the_rest(self, tmp_path):
        got = self._parse(tmp_path, build_jpeg(gps=False))
        assert got["camera"] == "Google Pixel 8"
        assert got["taken"].startswith("2026:07:14")
        assert "lat" not in got

    def test_the_rest_of_what_a_phone_writes(self, tmp_path):
        """Everything else in the file, not just the six tags it started with."""
        got = self._parse(tmp_path, build_jpeg(rich=True))
        assert got["offset"] == "+01:00", "the time zone, without which elapsed time is a guess"
        assert got["utc"].startswith("2026:07:14 17:42"), "GPS keeps UTC"
        assert got["lens"] == "Pixel 8 back camera"
        assert (got["width"], got["height"]) == (4080, 3072)
        assert got["exposure"] == "1/120 s", "written the way a shutter speed is read"
        assert got["aperture"] == "f/1.8"
        assert got["focal"] == "6.7 mm"
        assert got["focal35"] == "24 mm"
        assert got["iso"] == 64
        assert got["flash"] is True
        assert got["accuracy"] == 8
        assert got["speed"] == "12.3 km/h"
        assert got["heading"] == "274°"
        assert got["orientation"] == "rotated 90° clockwise"
        assert got["software"] == "HDR+ 1.0"
        assert got["artist"] == "tj"

    @pytest.mark.parametrize("shape", [
        {"jfif": True},
        {"fill_bytes": True},
        {"thumbnail": True},
        {"jfif": True, "fill_bytes": True, "thumbnail": True},
    ])
    def test_the_shape_a_real_file_has_around_the_exif(self, tmp_path, shape):
        """A JFIF segment first, padding before a marker, a thumbnail IFD after.

        The padding is the one that bit: any number of 0xFF bytes may sit before
        a marker and the standard says to skip them. Reading one as the marker
        gave a nonsense segment length and lost the whole file — every tag, not
        just the padded segment.
        """
        got = self._parse(tmp_path, build_jpeg(rich=True, **shape))
        assert got is not None, shape
        assert got["camera"] == "Google Pixel 8"
        assert got["lat"] == pytest.approx(51.51, abs=1e-6)

    def test_a_plain_photo_reports_only_what_it_has(self, tmp_path):
        """The extra tags are absent, not empty — nothing is invented."""
        got = self._parse(tmp_path, build_jpeg())
        for key in ("offset", "utc", "lens", "iso", "flash", "orientation",
                    "software", "artist", "accuracy", "speed", "heading"):
            assert key not in got, key

    @pytest.mark.parametrize("data, why", [
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "a PNG has no APP1 at all"),
        (b"\xff\xd8\xff\xe1\x00\x10Exif\x00\x00" + b"\xff" * 200, "truncated garbage"),
        (b"\xff\xd8", "nothing but a start marker"),
        (b"", "an empty file"),
    ])
    def test_anything_that_is_not_exif_returns_null_without_throwing(self, tmp_path, data, why):
        assert self._parse(tmp_path, data) is None, why

    def test_an_ifd_pointer_past_the_end_is_survived(self, tmp_path):
        """The input a hand-rolled parser gets wrong: a hostile offset."""
        bad = bytearray(build_jpeg())
        bad[12 + 4:12 + 8] = struct.pack(">I", 0x7FFFFF00)   # IFD0 offset in the TIFF header
        assert self._parse(tmp_path, bytes(bad)) is None

    @pytest.mark.parametrize("keep", [24, 40, 56, 80, 120, 180, 240])
    def test_a_truncated_file_reads_what_survived_and_no_further(self, tmp_path, keep):
        """Every value offset now points past the buffer. None may be read.

        This is what a partial upload looks like, and it is the case where a
        parser that trusts its own offsets reads adjacent memory or throws.
        """
        got = self._parse(tmp_path, build_jpeg()[:keep])
        assert got is None or isinstance(got, dict)
        if isinstance(got, dict):
            # Anything it did return has to be genuinely present, not garbage
            # scraped from beyond the end.
            assert set(got) <= {"camera", "taken", "lat", "lon", "altitude"}
            if "camera" in got:
                assert got["camera"] in ("Google Pixel 8", "Google", "Pixel 8")


# --------------------------------------------------------------------------
# The prose it becomes
# --------------------------------------------------------------------------

FULL = {"camera": "Google Pixel 8", "taken": "2026:07:14 18:42:07",
        "lat": 51.51, "lon": -0.1275, "altitude": 42}


class TestRenderingTheFacts:
    def test_a_single_photo_reads_as_a_sentence(self):
        out = web.image_metadata([FULL])
        assert "- Photo: " in out
        assert "Tuesday 14 July 2026 at 18:42 (evening)" in out
        assert "51.510000, -0.127500" in out
        assert "42 m above sea level" in out
        assert "on a Google Pixel 8" in out

    def test_several_photos_are_numbered(self):
        out = web.image_metadata([FULL, FULL])
        assert "- Image 1:" in out and "- Image 2:" in out
        assert "- Photo:" not in out

    @pytest.mark.parametrize("entries", [[], [None], [{}], [None, {}], None])
    def test_nothing_known_produces_nothing_at_all(self, entries):
        assert web.image_metadata(entries) == ""

    def test_a_photo_with_only_a_time_says_only_the_time(self):
        out = web.image_metadata([{"taken": "2026:01:03 07:15:00"}])
        assert "morning" in out
        assert "," not in out.split("- Photo: ")[1], "no empty trailing clauses"

    @pytest.mark.parametrize("hour, part", [
        ("02", "night"), ("09", "morning"), ("14", "afternoon"), ("21", "evening"),
    ])
    def test_the_time_of_day_is_named(self, hour, part):
        assert part in web.image_metadata([{"taken": f"2026:07:14 {hour}:00:00"}])

    @pytest.mark.parametrize("raw", ["", "not a date", "9999:99:99 99:99", None, 12345])
    def test_an_unparseable_timestamp_never_raises(self, raw):
        web.image_metadata([{"taken": raw, "camera": "X"}])   # must not throw

    def test_altitude_without_a_position_is_not_claimed(self):
        """Metres above sea level with no coordinates is not a location."""
        out = web.image_metadata([{"altitude": 42, "camera": "X"}])
        assert "sea level" not in out

    RICH = dict(FULL, offset="+01:00", utc="2026:07:14 17:42:07",
                lens="Pixel 8 back camera", width=4080, height=3072,
                exposure="1/120 s", aperture="f/1.8", focal="6.7 mm",
                focal35="24 mm", iso=64, flash=True, accuracy=8.0,
                speed="12.3 km/h", heading="274°",
                orientation="rotated 90° clockwise", software="HDR+ 1.0")

    def test_everything_the_file_carried_is_said(self):
        out = web.image_metadata([self.RICH])
        for fragment in ("(UTC+01:00)", "give or take 8 m", "moving at 12.3 km/h",
                         "facing 274°", "(Pixel 8 back camera)", "4080×3072 (13 MP)",
                         "1/120 s", "f/1.8", "6.7 mm", "24 mm equivalent", "ISO 64",
                         "flash fired", "held rotated 90° clockwise",
                         "written by HDR+ 1.0"):
            assert fragment in out, fragment

    def test_the_time_zone_is_said_where_it_is_known(self):
        """Without it, two photos across a zone change are hours apart in a way
        nothing downstream can detect — which the Trip routine warns about."""
        assert "(UTC+01:00)" in web.image_metadata([dict(FULL, offset="+01:00")])
        assert "UTC" not in web.image_metadata([FULL]).split("- Photo:")[1]

    def test_the_gps_clock_stands_in_for_a_missing_offset(self):
        """GPS keeps UTC; beside a local capture time that is the zone."""
        out = web.image_metadata([dict(FULL, utc="2026:07:14 17:42:07")])
        assert "which was 2026:07:14 17:42:07 UTC" in out

    def test_the_offset_wins_when_both_are_there(self):
        out = web.image_metadata([dict(FULL, offset="+01:00", utc="2026:07:14 17:42:07")])
        assert "(UTC+01:00)" in out
        assert "which was" not in out, "saying it twice adds nothing"

    def test_the_settings_are_skipped_when_absent(self):
        out = web.image_metadata([FULL])
        for fragment in ("ISO", "f/", "flash", "MP", "equivalent"):
            assert fragment not in out, fragment

    def test_no_flash_is_said_as_plainly_as_flash(self):
        assert "no flash" in web.image_metadata([dict(FULL, flash=False)])
        assert "flash fired" in web.image_metadata([dict(FULL, flash=True)])

    def test_a_lens_without_a_camera_still_gets_a_mention(self):
        out = web.image_metadata([{"lens": "24mm prime", "taken": "2026:07:14 18:42:07"}])
        assert "with a 24mm prime" in out

    def test_the_accuracy_qualifies_the_position(self):
        """Six decimal places reads like a doorstep and can be a whole block."""
        out = web.image_metadata([dict(FULL, accuracy=45.0)])
        assert "51.510000, -0.127500 (give or take 45 m)" in out

    def test_speed_and_heading_go_with_the_position(self):
        """They are GPS facts; without a fix they describe nothing."""
        out = web.image_metadata([{"speed": "12 km/h", "heading": "90°",
                                   "taken": "2026:07:14 18:42:07"}])
        assert "moving at" not in out and "facing" not in out

    def test_a_hostile_field_cannot_run_off(self):
        long_lens = "L" * 400
        out = web.image_metadata([dict(FULL, lens=long_lens, software="S" * 400)])
        assert long_lens not in out and "S" * 400 not in out

    def test_the_preamble_marks_it_as_data(self):
        out = web.image_metadata([FULL])
        assert "not instructions" in out, "the injection fence must be here too"

    @pytest.mark.parametrize("meta", [
        {"lat": float("inf"), "lon": 0.0},
        {"lat": float("nan"), "lon": 0.0},
        {"lat": 1.0, "lon": 2.0, "altitude": float("inf")},
        {"lat": 1.0, "lon": 2.0, "altitude": float("nan")},
        {"lat": 900.0, "lon": 2.0},                 # not a latitude
        {"lat": 1.0, "lon": 4000.0},                # not a longitude
        {"lat": "51.5", "lon": "-0.1"},             # strings, not numbers
        {"lat": True, "lon": False},                # bools are ints in Python
    ])
    def test_an_impossible_coordinate_is_dropped_not_printed(self, meta):
        """json.loads accepts Infinity and NaN, so a request body can carry them."""
        out = web.image_metadata([dict(meta, camera="X")])
        for bad in ("inf", "nan", "900.0", "4000.0", "51.5"):
            assert bad not in out
        assert "sea level" not in out

    def test_a_real_coordinate_still_gets_through(self):
        assert "51.510000" in web.image_metadata([{"lat": 51.51, "lon": -0.1275}])

    def test_null_island_is_not_a_place_a_photo_was_taken(self):
        """A camera writes 0/0/0 for a GPS block it never got a fix for.

        Reported as a position it becomes the Gulf of Guinea, and the model
        answers "off the coast of Ghana" with complete confidence.
        """
        out = web.image_metadata([{"lat": 0.0, "lon": 0.0, "camera": "X"}])
        assert "0.000000" not in out
        assert "on a X" in out, "the rest of the photo's facts still stand"

    @pytest.mark.parametrize("lat, lon", [(0.0, 12.5), (12.5, 0.0), (0.0, -0.0001)])
    def test_a_real_position_on_a_zero_meridian_survives(self, lat, lon):
        """Only exactly 0,0 is the no-fix marker; Greenwich is a real place."""
        assert "at " in web.image_metadata([{"lat": lat, "lon": lon}])

    def test_a_hostile_camera_name_cannot_run_off(self):
        long_name = "A" * 500
        out = web.image_metadata([{"camera": long_name}])
        assert long_name not in out and "A" * 60 in out


# --------------------------------------------------------------------------
# Picking it out of a conversation
# --------------------------------------------------------------------------

class TestFindingItInAThread:
    def test_it_comes_from_the_turn_the_images_came_from(self):
        messages = [
            {"role": "user", "content": "look", "images": ["a"], "image_meta": [FULL]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "and again?"},
        ]
        assert web.conversation_image_meta(messages) == [FULL]

    def test_a_newer_photo_without_metadata_hides_an_older_one(self):
        """Entry n must describe photo n — never a previous turn's photo."""
        messages = [
            {"role": "user", "content": "one", "images": ["a"], "image_meta": [FULL]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "two", "images": ["b"]},
        ]
        assert web.conversation_image_meta(messages) == []

    def test_it_lines_up_with_the_images_positionally(self):
        messages = [{"role": "user", "content": "x", "images": ["a", "b"],
                     "image_meta": [None, FULL]}]
        assert web.conversation_image_meta(messages) == [None, FULL]

    @pytest.mark.parametrize("meta", ["nonsense", 7, {"lat": 1}, [1, 2], [[]]])
    def test_a_malformed_payload_is_ignored_rather_than_trusted(self, meta):
        messages = [{"role": "user", "content": "x", "images": ["a"], "image_meta": meta}]
        assert web.conversation_image_meta(messages) == []

    def test_a_thread_with_no_images_has_nothing_to_find(self):
        assert web.conversation_image_meta([{"role": "user", "content": "hi"}]) == []
        assert web.conversation_image_meta([]) == []
        assert web.conversation_image_meta(None) == []

    def test_stripping_leaves_the_rest_of_the_message_alone(self):
        messages = [{"role": "user", "content": "x", "images": ["a"], "image_meta": [FULL]}]
        out = web.strip_image_meta(messages)
        assert out[0] == {"role": "user", "content": "x", "images": ["a"]}
        assert messages[0]["image_meta"] == [FULL], "the caller's list must not change"


# --------------------------------------------------------------------------
# The whole path
# --------------------------------------------------------------------------

class _Ollama:
    """Records what the server actually sends, and streams a short reply."""

    def __init__(self):
        self.calls = []

    def stream(self, model, messages, options=None, keep_alive=None, **kw):
        self.calls.append({"model": model, "messages": messages, "stream": True})
        yield json.dumps({"message": {"content": "ok"}, "done": True})

    def blocking(self, model, messages, **kw):
        self.calls.append({"model": model, "messages": messages, "stream": False})
        return "NONE"


@pytest.fixture
def rig(monkeypatch):
    fake = _Ollama()
    monkeypatch.setattr(app_module, "chat_stream", fake.stream)
    monkeypatch.setattr(app_module, "ollama_chat", fake.blocking)
    monkeypatch.setattr(app_module, "_model_has_vision", lambda m: True)
    return {"ollama": fake, "client": app_module.app.test_client()}


def _sent(fake) -> list:
    streamed = [c for c in fake.calls if c["stream"]]
    assert streamed, "no chat call was made"
    return streamed[-1]["messages"]


class TestItReachesTheModel:
    def test_the_facts_arrive_as_a_system_turn(self, rig):
        resp = rig["client"].post("/api/chat", json={"messages": [
            {"role": "user", "content": "where was this?", "images": ["ZmFrZQ=="],
             "image_meta": [FULL]}]})
        resp.get_data()
        sent = _sent(rig["ollama"])
        assert sent[0]["role"] == "system"
        assert "51.510000" in sent[0]["content"]
        assert "Tuesday 14 July 2026" in sent[0]["content"]

    def test_the_raw_payload_does_not_ride_along(self, rig):
        resp = rig["client"].post("/api/chat", json={"messages": [
            {"role": "user", "content": "where?", "images": ["ZmFrZQ=="],
             "image_meta": [FULL]}]})
        resp.get_data()
        for msg in _sent(rig["ollama"]):
            assert "image_meta" not in msg, "it is sent once, in prose, not as JSON"

    def test_a_photo_with_no_metadata_adds_no_system_turn(self, rig):
        resp = rig["client"].post("/api/chat", json={"messages": [
            {"role": "user", "content": "what is this?", "images": ["ZmFrZQ=="]}]})
        resp.get_data()
        assert [m["role"] for m in _sent(rig["ollama"])] == ["user"]

    def test_a_text_only_model_gets_it_alongside_the_transcription(self, rig, monkeypatch):
        """The OCR branch rebuilds `convo` from scratch — after, not before."""
        monkeypatch.setattr(app_module, "_model_has_vision", lambda m: False)
        monkeypatch.setattr(app_module, "_image_reader", lambda m: ("reader:1b", True))
        monkeypatch.setattr(web, "read_images", lambda *a, **k: "SETTINGS  Wi-Fi")
        resp = rig["client"].post("/api/chat", json={"messages": [
            {"role": "user", "content": "where was this?", "images": ["ZmFrZQ=="],
             "image_meta": [FULL]}]})
        resp.get_data()
        system = "\n".join(m["content"] for m in _sent(rig["ollama"]) if m["role"] == "system")
        assert "SETTINGS" in system, "the transcription was lost"
        assert "51.510000" in system, "the metadata was overwritten by the OCR context"

    def test_the_non_streaming_path_strips_it_too(self, rig):
        rig["client"].post("/api/chat", json={"stream": False, "messages": [
            {"role": "user", "content": "hi", "images": ["ZmFrZQ=="], "image_meta": [FULL]}]})
        blocking = [c for c in rig["ollama"].calls if not c["stream"]][-1]
        assert all("image_meta" not in m for m in blocking["messages"])


class TestTheSearchPlannerCanUseIt:
    """"What's this building?" is a different search from a known position.

    The planner runs on the same machine as everything else, so it may have the
    photo's facts — but what it *writes* is sent to a search engine, which makes
    this the one place the position can leave the house. Hence its own switch.
    """

    def _run(self, rig, monkeypatch, **env):
        monkeypatch.setenv("WEB_ENABLED", "1")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        seen, searched = {"planner": [], "given": None}, []
        monkeypatch.setattr(web, "search", lambda q, limit=3: searched.append(q) or [])

        def planner(model, messages, **kw):
            seen["planner"].append(json.dumps(messages))
            return "Q: nearby landmarks"

        # The planner imports ollama_client.chat itself, so patching app's
        # reference does not reach it.
        monkeypatch.setattr("ollama_client.chat", planner)

        real = app_module._gather_web

        def spy(model, messages, *a, **kw):
            seen["given"] = json.dumps(messages)
            return real(model, messages, *a, **kw)

        monkeypatch.setattr(app_module, "_gather_web", spy)
        resp = rig["client"].post("/api/chat", json={"web": True, "messages": [
            {"role": "user", "content": "what is near here?", "images": ["ZmFrZQ=="],
             "image_meta": [FULL]}]})
        resp.get_data()
        assert seen["given"] is not None, "the web path did not run"
        assert seen["planner"], "the planner did not run"
        return seen, searched

    def test_the_planner_is_told_where_and_when(self, rig, monkeypatch):
        seen, _ = self._run(rig, monkeypatch)
        prompt = seen["planner"][0]
        assert "51.510000" in prompt and "-0.127500" in prompt
        assert "14 July 2026" in prompt

    def test_a_long_note_is_cut_at_a_separator_not_mid_fact(self):
        """"140 m above" reads like a fact the photo recorded. It is half of one."""
        two = [{"taken": "2026:08:07 09:04:00", "lat": 51.51, "lon": -0.1275,
                "altitude": 42, "camera": "Google Pixel 8"},
               {"taken": "2026:08:07 17:32:00", "lat": 52.48, "lon": -1.9025,
                "altitude": 140, "camera": "Google Pixel 8"}]
        note = web.metadata_note(two)
        assert len(note) <= 220
        assert not note.endswith("above")
        assert note.rstrip().endswith(("sea level", "Pixel 8", ")")) or ", " not in note[-25:], note

    def test_a_note_that_fits_is_left_whole(self):
        note = web.metadata_note([{"taken": "2026:08:07 09:04:00"}])
        assert note.endswith("(morning)")

    def test_it_arrives_as_a_note_not_as_the_fenced_system_turn(self, rig, monkeypatch):
        """The planner gets a few hundred characters; a preamble would eat them."""
        seen, _ = self._run(rig, monkeypatch)
        prompt = seen["planner"][0]
        assert "the photo records:" in prompt
        assert "not instructions" not in prompt, "the long preamble must not be here"

    def test_the_conversation_itself_still_carries_no_raw_payload(self, rig, monkeypatch):
        """It travels as prose put somewhere deliberately, not as a JSON field."""
        seen, _ = self._run(rig, monkeypatch)
        assert "image_meta" not in seen["given"]

    def test_the_switch_keeps_the_date_and_drops_the_position(self, rig, monkeypatch):
        seen, _ = self._run(rig, monkeypatch, WEB_SHARE_LOCATION="0")
        prompt = seen["planner"][0]
        assert "51.510000" not in prompt and "-0.127500" not in prompt
        assert "14 July 2026" in prompt, "the date is not the sensitive part"
        assert "Google Pixel 8" in prompt

    def test_with_the_switch_off_no_coordinate_reaches_the_search_engine(self, rig, monkeypatch):
        _, searched = self._run(rig, monkeypatch, WEB_SHARE_LOCATION="0")
        assert searched, "nothing was searched"
        for query in searched:
            assert "51.51" not in query and "0.1275" not in query

    def test_the_answering_model_gets_the_full_version_regardless(self, rig, monkeypatch):
        self._run(rig, monkeypatch, WEB_SHARE_LOCATION="0")
        system = [m for m in _sent(rig["ollama"]) if m["role"] == "system"]
        assert any("51.510000" in m["content"] for m in system), \
            "withholding it from search must not withhold it from the answer"

    def test_a_stale_photo_stops_steering_later_searches(self, rig, monkeypatch):
        """Same gate as the transcription: only an image attached to this turn."""
        monkeypatch.setenv("WEB_ENABLED", "1")
        monkeypatch.setattr(web, "search", lambda q, limit=3: [])
        prompts = []
        monkeypatch.setattr("ollama_client.chat",
                            lambda model, messages, **kw: prompts.append(json.dumps(messages))
                            or "Q: paris weather")
        resp = rig["client"].post("/api/chat", json={"web": True, "messages": [
            {"role": "user", "content": "what is this?", "images": ["ZmFrZQ=="],
             "image_meta": [FULL]},
            {"role": "assistant", "content": "A building."},
            {"role": "user", "content": "what is the weather in Paris?"},
        ]})
        resp.get_data()
        assert prompts, "the planner did not run"
        assert "51.510000" not in prompts[0]


class TestTheDeploymentPicksTheDefault:
    """A fresh browser follows the server; one that has chosen keeps its choice."""

    def _health(self, rig, monkeypatch, **env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(app_module, "list_models", lambda **kw: [])
        return rig["client"].get("/api/health").get_json()

    def test_on_by_default(self, rig, monkeypatch):
        assert self._health(rig, monkeypatch)["photo_meta"] is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_the_operator_can_turn_it_round(self, rig, monkeypatch, value):
        assert self._health(rig, monkeypatch, PHOTO_META=value)["photo_meta"] is False

    def test_the_page_starts_it_off_and_lets_health_turn_it_on(self):
        """Off first, then on — never the reverse.

        A page that started it on and waited to be corrected would read a
        photo's position anyway in the window before /api/health answered,
        which is exactly what PHOTO_META=0 is meant to prevent.
        """
        page = chat_ui.render_page("t")
        assert 'rememberToggle(exifEl, "chatExif", false)' in page, \
            "the page's own default must be off"
        # The server's answer is applied inside checkVoice, after the bar is
        # shown and before anything else in that response is acted on.
        window = page[page.index("exifBar.hidden = false"):
                      page.index("if (data.history)")]
        assert "data.photo_meta" in window, "health must apply the default here"
        assert 'chosen("chatExif")' in window, "a real choice must not be overridden"
        assert "pendingRoutine" in window, \
            "a tab resume must not lower the toggle underneath an armed routine"


# --------------------------------------------------------------------------
# Surviving a reload
# --------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    return tmp_path / "chat.db"


class TestItSurvivesReopeningTheThread:
    def test_it_round_trips(self, db):
        convo = store.create("photos")
        assert store.add_message(convo["id"], "user", "where?", ["b64"], None, [FULL])
        back = store.get(convo["id"])["messages"][0]
        assert back["image_meta"] == [FULL]

    def test_a_message_without_any_stores_null(self, db):
        convo = store.create("t")
        store.add_message(convo["id"], "user", "hi", ["b64"], None, [None])
        assert store.get(convo["id"])["messages"][0]["image_meta"] is None

    def test_an_older_database_gains_the_column(self, db, monkeypatch):
        """The upgrade path: a chat.db written before this feature existed."""
        import sqlite3
        conn = sqlite3.connect(str(db))
        conn.executescript(store._SCHEMA.replace("    image_meta      TEXT,\n", ""))
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at)"
            " VALUES ('old', 'before', 1.0, 1.0)")
        conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, created_at)"
            " VALUES ('old', 1, 'user', 'hello', 1.0)")
        conn.commit()
        conn.close()

        # Reading must work, and so must writing a message that has metadata.
        assert store.get("old")["messages"][0]["image_meta"] is None
        assert store.add_message("old", "user", "and this?", ["b64"], None, [FULL])
        assert store.get("old")["messages"][1]["image_meta"] == [FULL]


class TestOrientationCannotMisleadTheModel:
    """The pixels are turned upright before sending; the tag says they were not.

    loadBitmap asks for imageOrientation "from-image", so the model receives an
    upright image. Reporting "rotated 90° clockwise" on its own reads as a claim
    about that image, and a model that believes it will try to compensate for a
    rotation already undone — which is precisely the odometer-OCR case.
    """

    def test_it_is_said_about_the_camera_not_the_picture(self):
        out = web.image_metadata([{"taken": "2026:07:14 18:42:07",
                                   "orientation": "rotated 90° clockwise"}])
        assert "camera held rotated 90° clockwise" in out
        assert "already turned upright here" in out

    def test_the_page_really_does_turn_them_upright(self):
        """If this ever stops being true, the sentence above becomes a lie."""
        page = chat_ui.render_page("t")
        assert 'imageOrientation: "from-image"' in page

    def test_an_upright_photo_says_nothing_about_it(self):
        """Orientation 1 is the default and the parser drops it."""
        out = web.image_metadata([{"taken": "2026:07:14 18:42:07"}])
        assert "held" not in out.split("- Photo:")[1]
