"""Getting the right sound, from the right microphone, into the recogniser.

"It transcribed it wrongly" has three quite different causes: it recorded from
the wrong device, it recorded nothing worth transcribing, or what it recorded
was mangled on the way to 16 kHz. The first two you can only rule out by being
shown them; the third is measurable, and was measurably bad.
"""

from __future__ import annotations

import json
import re
import subprocess

import pytest

import chat_ui
from conftest import page_script


def page() -> str:
    return chat_ui.render_page("t")


def run_js(extra: str) -> dict:
    """Run the page's own resampler under node with a measurement bolted on."""
    js = page_script(page())
    start = js.index("      const RESAMPLE_TAPS")
    end = js.index("      // Convert Float32 samples to 16 kHz")
    harness = r"""
    function tone(hz, rate, secs) {
      const n = Math.round(rate * secs), out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = Math.sin(2 * Math.PI * hz * i / rate);
      return out;
    }
    function rms(a) { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s / a.length); }
    function db(x) { return 20 * Math.log10(Math.max(x, 1e-12)); }
    // How much of a tone at `hz` survives the trip to 16 kHz, in dB.
    function keeps(hz, inRate) {
      const src = tone(hz, inRate, 0.5);
      const got = resample(src, inRate, 16000);
      return db(rms(got.subarray(400, got.length - 400)) /
                rms(src.subarray(400, src.length - 400)));
    }
    """
    out = subprocess.run(["node", "-e", js[start:end] + harness + extra],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class TestNothingAboveNyquistFoldsBackOntoTheSpeech:
    """Decimating without a filter first drops everything above 8 kHz onto the
    band underneath it — a 12 kHz sound lands at 4 kHz, on top of a vowel. The
    box average that used to do this job measures at -13 dB, which leaves most
    of it there.
    """

    @pytest.mark.parametrize("in_rate", [44100, 48000])
    def test_out_of_band_sound_is_rejected(self, in_rate):
        got = run_js(
            "console.log(JSON.stringify([9000, 10000, 12000, 16000, 20000]"
            f".map(hz => keeps(hz, {in_rate}))));")
        assert max(got) < -45, f"worst rejection {max(got):.1f} dB at {in_rate}"

    @pytest.mark.parametrize("in_rate", [44100, 48000])
    def test_and_the_speech_band_comes_through_untouched(self, in_rate):
        got = run_js(
            "console.log(JSON.stringify([100, 300, 1000, 2000, 3000, 4000, 5000]"
            f".map(hz => keeps(hz, {in_rate}))));")
        assert min(got) > -0.5, f"lost {min(got):.2f} dB of the speech band"
        assert max(got) < 0.5, "and nothing was amplified"

    def test_a_matching_rate_is_left_completely_alone(self):
        got = run_js("console.log(JSON.stringify("
                     "Array.from(resample(tone(1000, 16000, 0.01), 16000, 16000))"
                     ".slice(0, 4)));")
        expect = run_js("console.log(JSON.stringify("
                        "Array.from(tone(1000, 16000, 0.01)).slice(0, 4)));")
        assert got == expect

    def test_the_kernel_is_built_once_per_rate_pair(self):
        """A phone re-deriving 97 coefficients for every pause in a
        conversation is work for nothing."""
        js = page_script(page())
        at = js.index("function kernelFor")
        assert "if (resampleKernel && resampleKernel.ratio === ratio)" in js[at:at + 200]

    def test_it_is_flat_at_dc(self):
        """An un-normalised kernel quietly changes the recording's volume,
        which is the one thing the recogniser's own gain cannot undo."""
        got = run_js("const a = new Float32Array(4800).fill(0.5);"
                     "const b = resample(a, 48000, 16000);"
                     "console.log(JSON.stringify(b[800]));")
        assert abs(got - 0.5) < 0.001, f"DC gain came out at {got / 0.5:.4f}"

    def test_upsampling_is_guarded_rather_than_silently_wrong(self):
        """The kernel is a decimator. A mic below 16 kHz is not a thing this
        has ever seen, but producing something confidently wrong would be."""
        js = page_script(page())
        at = js.index("function resample(samples, inRate, outRate)")
        assert "if (ratio < 1)" in js[at:at + 700]


class TestYouCanSeeWhichMicrophoneItIsUsing:
    """The browser's default is chosen by the operating system and is regularly
    not the one being spoken into: a laptop offers the webcam's, a desktop
    offers whatever the monitor came with.
    """

    def test_there_is_a_picker(self):
        text = page()
        assert 'id="micDevice"' in text
        assert text.index('id="micDevice"') < text.index('id="voiceModel"')

    def test_a_chosen_device_is_demanded_rather_than_preferred(self):
        """ideal falls back in silence, which is how you end up recording from
        the webcam while the picker says headset."""
        js = page_script(page())
        at = js.index("const want = micDeviceEl.hidden")
        assert "constraints.deviceId = { exact: want };" in js[at:at + 700]

    def test_a_device_that_has_been_unplugged_falls_back_and_says_so(self):
        js = page_script(page())
        at = js.index("const want = micDeviceEl.hidden")
        window = js[at:at + 1400]
        assert "delete constraints.deviceId;" in window
        assert "not available" in window

    def test_the_choice_is_remembered_but_only_if_it_still_exists(self):
        js = page_script(page())
        assert 'remember("mic", micDeviceEl.value);' in js
        at = js.index("async function loadMicDevices")
        assert "mics.some(d => d.deviceId === want)" in js[at:at + 1600]

    def test_changing_it_takes_effect_immediately(self):
        """You change this because the last recording came from the wrong
        place; waiting for the next tap is waiting for another bad one."""
        js = page_script(page())
        at = js.index('micDeviceEl.addEventListener("change"')
        assert "if (recording) { await stopMic(); await startMic(); }" in js[at:at + 500]

    def test_a_device_name_is_never_written_as_markup(self):
        js = page_script(page())
        at = js.index("async function loadMicDevices")
        window = js[at:js.index('micDeviceEl.addEventListener("change"')]
        assert "opt.textContent = label;" in window
        assert re.findall(r"(\w+)\.innerHTML\s*=", window) == ["micDeviceEl"]

    def test_it_is_hidden_where_there_is_nothing_to_choose(self):
        """One nameless entry, which is what you get before permission, is a
        choice of one blank."""
        js = page_script(page())
        at = js.index("async function loadMicDevices")
        assert "if (mics.length < 2 && !mics.some(d => d.label))" in js[at:at + 1200]

    def test_it_refills_when_something_is_plugged_in(self):
        js = page_script(page())
        assert 'addEventListener("devicechange", loadMicDevices)' in js


class TestYouCanSeeWhetherItIsHearingYou:
    def test_there_is_a_level_meter(self):
        text = page()
        assert 'id="micLevel"' in text and 'id="micLevelBar"' in text

    def test_it_runs_in_both_modes(self):
        """The VAD only runs in continuous mode, and "am I being heard" is the
        question in both."""
        js = page_script(page())
        at = js.index("procNode.onaudioprocess")
        window = js[at:at + 700]
        assert window.index("showLevel(chunk);") < window.index("continuousEl.checked")

    def test_it_is_a_log_scale(self):
        """Speech at a sensible level sits near 0.05 linear, which on a linear
        bar is a sliver against a full-scale end."""
        js = page_script(page())
        at = js.index("function showLevel")
        assert "20 * Math.log10" in js[at:at + 900]

    def test_clipping_is_shown(self):
        js = page_script(page())
        at = js.index("function showLevel")
        assert 'classList.toggle("hot", peak > 0.98)' in js[at:at + 900]

    def test_the_peak_survives_the_moment_it_happened(self):
        js = page_script(page())
        at = js.index("function showLevel")
        assert "if (peak > micPeak) micPeak = peak;" in js[at:at + 900]

    def test_the_meter_only_shows_while_recording(self):
        js = page_script(page())
        assert "resetLevel(); micLevelEl.hidden = false;" in js
        assert "micLevelEl.hidden = true;" in js

    def test_silence_is_explained_rather_than_just_reported(self):
        """"No speech detected" with no reason sends people to the model
        picker when the problem is the microphone."""
        js = page_script(page())
        at = js.index("No speech detected.")
        window = js[max(0, at - 600):at + 200]
        assert "micPeak < 0.02" in window
        assert "barely registered" in window
        assert "clipping" in window

    def test_the_hint_names_the_source_it_is_recording_from(self):
        js = page_script(page())
        assert 'const from = micLabel ? " from " + micLabel : "";' in js
