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


def run_vad(extra: str):
    """Drive the page's own detector with synthetic rooms.

    The whole capture path from the high-pass down to what gets posted, with
    only the parts that talk to the browser or the network stubbed out.
    """
    js = page_script(page())
    start = js.index("      // Most of a room is below the speech.")
    # Through stopMic, so the decision about what to post is the real one
    # rather than one the harness re-implements and then agrees with.
    end = js.index("      // Post one utterance for transcription")
    harness = r"""
    let micRate = 16000, buffers = [], bufferedSamples = 0;
    let sent = [], recording = false, hpState = null, micStarting = false;
    // What stopMic touches on its way out. None of it is the point.
    const micBtn = { classList: { add() {}, remove() {} } };
    const micLevelEl = {}, hintEl = {}, inputEl = { focus() {} };
    const continuousEl = { checked: true }, autoSendEl = { checked: false };
    const voiceSel = { value: "" }, headsetEl = { checked: true };
    const micDeviceEl = { hidden: true, value: "" };
    let procNode = { disconnect() {} }, srcNode = { disconnect() {} };
    let audioCtx = { close: async () => {} }, mediaStream = { getTracks: () => [] };
    // Seeded, not Math.random(): a test that fails one run in twenty is worse
    // than no test, and "it passed when I ran it" is not a measurement.
    let seed = 12345;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
    function mergeBuffers(list) {
      let n = 0; for (const b of list) n += b.length;
      const o = new Float32Array(n); let k = 0;
      for (const b of list) { o.set(b, k); k += b.length; }
      return o;
    }
    function encodeWav(s) { return s.length ? s : null; }
    function transcribeBlob(w) { sent.push(w.length / micRate); }
    let stopped = 0;
    // The idle stop schedules itself; count it, then let the real one run.
    setTimeout = (fn) => { stopped++; };
    const micLevelBarEl = { style: {}, classList: { toggle() {} } };
    const micLevelPeakEl = { style: {} };

    function noise(n, amp) {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = (rnd() * 2 - 1) * amp;
      return a;
    }
    function hum(n, hz, amp, ph) {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = Math.sin(2 * Math.PI * hz * (i + ph) / micRate) * amp;
      return a;
    }
    // A voice: harmonics of a wandering pitch, tilted the way a real one is,
    // with formant-ish bumps and a syllable-rate envelope. A pure tone would
    // pass a test that real speech fails.
    function voice(n, amp, ph) {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const t = (i + ph) / micRate;
        const f0 = 130 + 25 * Math.sin(2 * Math.PI * 0.7 * t);
        let v = 0;
        for (let h = 1; h <= 30; h++) {
          const f = f0 * h;
          v += Math.sin(2 * Math.PI * f * t) / Math.pow(h, 1.1)
               * (1 + 1.5 * Math.exp(-Math.pow((f - 700) / 400, 2))
                    + 1.0 * Math.exp(-Math.pow((f - 1800) / 700, 2)));
        }
        a[i] = v * (0.5 + 0.5 * Math.sin(2 * Math.PI * 4 * t)) * amp * 0.3;
      }
      return a;
    }
    function mix() {
      const xs = [...arguments], n = xs[0].length, o = new Float32Array(n);
      for (const x of xs) for (let i = 0; i < n; i++) o[i] += x[i];
      return o;
    }
    const ROOM = {
      quiet:  (ph) => noise(CH, 0.002),
      fan:    (ph) => mix(noise(CH, 0.010), hum(CH, 55, 0.020, ph), hum(CH, 120, 0.010, ph)),
      street: (ph) => mix(noise(CH, 0.014), hum(CH, 40, 0.035, ph), hum(CH, 90, 0.015, ph)),
      telly:  (ph) => mix(noise(CH, 0.020), voice(CH, 0.05, ph + 9999)),
    };
    const CH = 4096;
    // One recording, chunk by chunk, the way onaudioprocess sees it.
    async function record(room, amp, windows, seconds, continuous) {
      vadReset(); sent = []; stopped = 0; buffers = []; bufferedSamples = 0; seed = 12345;
      continuousEl.checked = continuous !== false;
      const hp = makeHighPass(micRate);
      const n = Math.round(seconds * micRate / CH);
      for (let c = 0; c < n; c++) {
        const ph = c * CH, t = ph / micRate;
        let raw = ROOM[room](ph);
        if (windows.some(w => t >= w[0] && t < w[1])) raw = mix(raw, voice(CH, amp, ph));
        const chunk = highPass(hp, raw);
        buffers.push(chunk); bufferedSamples += chunk.length;
        if (continuous !== false) vadStep(chunk);
      }
      const held = bufferedSamples;
      // And now the real stopMic decides what, if anything, gets posted.
      const before = sent.length;
      recording = true;
      await stopMic();
      return { sent: sent.slice(0, before), stopped,
               heldSecs: held / micRate,
               tailSecs: sent.length > before ? sent[before] : null,
               nothingHeard: sent.length === before };
    }
    """
    body = "(async () => {\n" + extra + "\n})();"
    out = subprocess.run(["node", "-e", js[start:end] + harness + body],
                         capture_output=True, text=True)
    if out.returncode:
        raise AssertionError(out.stderr[-2000:])
    return json.loads(out.stdout)


class TestMostOfARoomIsBelowTheSpeech:
    """A fan, a fridge, traffic, mains hum, a hand on the desk: nearly all of a
    room's energy is under 100 Hz, and none of it is speech. It was setting the
    threshold that speech then had to clear.
    """

    def keeps(self, hz):
        return run_js_hp(f"console.log(JSON.stringify(keeps({hz})));")

    @pytest.mark.parametrize("hz, floor", [(30, -35), (50, -20), (60, -15)])
    def test_the_rumble_goes(self, hz, floor):
        assert self.keeps(hz) < floor, f"{hz} Hz only came down {self.keeps(hz):.1f} dB"

    @pytest.mark.parametrize("hz", [150, 200, 300, 700, 1000, 3000, 6000])
    def test_and_the_speech_does_not(self, hz):
        """A low male fundamental sits near 100 Hz, but the formants the
        recogniser works from are all above this."""
        assert abs(self.keeps(hz)) < 0.5, f"{hz} Hz moved {self.keeps(hz):.2f} dB"

    def test_it_is_the_filter_it_claims_to_be(self):
        """Two biquads with the wrong pair of Q values is a different filter
        with a bump in the passband, which is worse than no filter at all."""
        peak = run_js_hp(
            "console.log(JSON.stringify(Math.max(...[150, 200, 300, 500, 800, "
            "1200, 2000, 4000].map(keeps))));")
        assert peak < 0.1, f"passband peaks at +{peak:.2f} dB"

    def test_it_carries_its_state_between_buffers(self):
        """Restarted every 4096 samples it rings at every boundary, which is a
        click at 4 Hz through the whole recording."""
        js = page_script(page())
        at = js.index("procNode.onaudioprocess")
        assert "hpState = makeHighPass(micRate);" in js[max(0, at - 300):at]
        assert "highPass(hpState, raw)" in js[at:at + 900]


def run_js_hp(extra: str):
    js = page_script(page())
    start = js.index("      const HP_HZ")
    end = js.index("      // Silence detection")
    harness = r"""
    const RATE = 16000;
    function rms(a) { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s / a.length); }
    // What the filter does to a steady tone, in dB. The first 0.2 s is skipped
    // because the filter has to settle.
    function keeps(hz) {
      const n = RATE, src = new Float32Array(n);
      for (let i = 0; i < n; i++) src[i] = Math.sin(2 * Math.PI * hz * i / RATE);
      const got = highPass(makeHighPass(RATE), src);
      return 20 * Math.log10(Math.max(rms(got.subarray(3200)) / rms(src.subarray(3200)), 1e-12));
    }
    """
    out = subprocess.run(["node", "-e", js[start:end] + harness + extra],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class TestSpeechInARoomWithSomethingRunningInIt:
    """Measured before this: a fan and mains hum put the floor at 0.017 with
    89% of that energy below 130 Hz. The threshold that follows is 0.050, and
    speech in the same room reached 0.023 — so nothing was ever detected and
    nothing was ever sent. The room, not the voice, decided.
    """

    # The quietest voice each room will let through, measured. Before the
    # high-pass and the 2x threshold these were 0.24 for a fan, 0.48 for a
    # street and 0.24 for a television — you had to raise your voice to be
    # heard at all, and next to traffic you had to shout.
    #
    # A television is the one that stays hard, and honestly so: it is broadband
    # and speech-shaped, so nothing short of telling two voices apart
    # distinguishes it from you. You do have to talk over it.
    @pytest.mark.parametrize("room, amp", [
        ("quiet", 0.03), ("fan", 0.08), ("street", 0.08), ("telly", 0.12),
    ])
    def test_a_sentence_is_heard(self, room, amp):
        got = run_vad(f"console.log(JSON.stringify(await record({room!r}, {amp}, [[2, 5]], 12, true)));")
        assert got["sent"], f"nothing was sent from a {room} room"

    @pytest.mark.parametrize("room", ["quiet", "fan", "street", "telly"])
    def test_and_an_empty_room_is_not(self, room):
        """The other half of it. A detector that hears everything is a
        detector that posts a minute of fan noise and lets the recogniser
        find words in it."""
        got = run_vad(f"console.log(JSON.stringify(await record({room!r}, 0, [], 12, true)));")
        assert got["sent"] == [], f"a {room} room on its own was sent"
        assert got["nothingHeard"] is True

    @pytest.mark.parametrize("room, was", [("fan", 0.16), ("street", 0.32), ("telly", 0.16)])
    def test_and_the_voice_that_used_to_be_needed_is_not(self, room, was):
        """The level that was still inaudible before. Three times quieter next
        to a fan, six times next to traffic — which is the difference between
        dictating and shouting at your phone.
        """
        got = run_vad(f"console.log(JSON.stringify(await record({room!r}, {was}, [[2, 5]], 12, true)));")
        assert got["sent"], f"{was} in a {room} room is still not heard"

    def test_two_sentences_are_two_utterances(self):
        got = run_vad("console.log(JSON.stringify("
                      "await record('fan', 0.08, [[2, 5], [7, 10]], 13, true)));")
        assert len(got["sent"]) == 2, got["sent"]

    def test_a_door_slamming_is_not_a_sentence(self):
        """Loud, short and broadband. This is what a threshold set low enough
        to hear a whisper starts posting."""
        got = run_vad(r"""
          const slam = (ph) => { const a = ROOM.fan(ph);
            for (let i = 0; i < CH; i++) { const k = ph + i - 2 * micRate;
              if (k >= 0 && k < micRate * 0.15)
                a[i] += (rnd() * 2 - 1) * 0.5 * Math.exp(-k / (micRate * 0.03)); }
            return a; };
          ROOM.slam = slam;
          console.log(JSON.stringify(await record('slam', 0, [], 12, true)));""")
        assert got["sent"] == []

    def test_and_a_sentence_after_one_still_is(self):
        got = run_vad(r"""
          ROOM.slam = (ph) => { const a = ROOM.fan(ph);
            for (let i = 0; i < CH; i++) { const k = ph + i - 2 * micRate;
              if (k >= 0 && k < micRate * 0.15)
                a[i] += (rnd() * 2 - 1) * 0.5 * Math.exp(-k / (micRate * 0.03)); }
            return a; };
          console.log(JSON.stringify(await record('slam', 0.08, [[6, 9]], 13, true)));""")
        assert len(got["sent"]) == 1, got["sent"]


class TestWhatGetsPostedIsTheSentence:
    """An utterance used to be "everything captured since the last one", which
    in a room the detector never triggered in is a minute of fan noise with
    three seconds of speech at the end. The recogniser does not ignore the
    rest — it looks for words in it, and finds some.
    """

    def test_the_room_before_the_sentence_is_not_sent(self):
        """Ten seconds of waiting, then three of talking."""
        got = run_vad("console.log(JSON.stringify("
                      "await record('quiet', 0.08, [[10, 13]], 16, true)));")
        assert len(got["sent"]) == 1
        assert 3.0 < got["sent"][0] < 4.2, f"{got['sent'][0]:.1f}s for a 3s sentence"

    def test_but_enough_of_it_to_not_clip_the_first_word(self):
        got = run_vad("console.log(JSON.stringify("
                      "await record('quiet', 0.08, [[10, 13]], 16, true)));")
        assert got["sent"][0] > 3.2, "no lead-in at all"

    def test_a_minute_of_nothing_is_never_posted(self):
        """The mic left on by accident. It used to upload the lot."""
        got = run_vad("console.log(JSON.stringify("
                      "await record('fan', 0, [], 70, true)));")
        assert got["sent"] == []
        assert got["nothingHeard"] is True
        assert got["stopped"] >= 1, "and the mic closes itself"

    def test_push_to_talk_sends_everything_it_recorded(self):
        """No detector runs, so there is no opinion to have: a deliberate tap
        on the mic is a deliberate request, silence and all."""
        got = run_vad("console.log(JSON.stringify("
                      "await record('quiet', 0.08, [[10, 13]], 16, false)));")
        assert got["nothingHeard"] is False
        assert got["tailSecs"] > 15, "push-to-talk was trimmed"

    def test_the_length_cap_applies_in_both_modes(self):
        """Continuous mode used to return before the check, so a mic in a room
        the detector never triggered in recorded until the upload 413'd."""
        js = page_script(page())
        at = js.index("procNode.onaudioprocess")
        window = js[at:at + 1200]
        assert "if (continuousEl.checked) vadStep(chunk);" in window
        assert window.index("bufferedSamples >= MAX_SAMPLES") > window.index("vadStep(chunk)")
        assert "return; }" not in window[:window.index("bufferedSamples >= MAX_SAMPLES")]


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
        window = js[at:at + 900]
        assert window.index("showLevel(raw);") < window.index("continuousEl.checked")

    def test_the_meter_reads_the_microphone_not_the_filter(self):
        """"Is it hearing me" and "is it clipping" are questions about the
        input. Clipping in particular has already happened by the time the
        page sees it, and a high-pass would hide the rumble that caused it."""
        js = page_script(page())
        at = js.index("procNode.onaudioprocess")
        window = js[at:at + 900]
        assert "showLevel(raw);" in window
        assert window.index("showLevel(raw);") < window.index("highPass(hpState, raw)")

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
        at = js.index("function nothingHeardHint")
        window = js[at:at + 600]
        assert "micPeak < 0.02" in window
        assert "barely registered" in window
        assert "clipping" in window

    def test_and_explained_the_same_way_wherever_it_is_reported(self):
        """Two places say it: the recogniser coming back empty, and the
        detector never having heard anything worth sending. One of them
        saying only "No speech detected." is the version you get when you are
        holding the phone wrong, which is when the reason matters most."""
        code = "\n".join(line for line in page_script(page()).split("\n")
                         if not line.strip().startswith("//"))
        spoken = re.findall(r'"No speech detected[^"]*"', code)
        assert spoken == ['"No speech detected."'], spoken
        calls = code.count("nothingHeardHint()") - code.count("function nothingHeardHint()")
        assert calls == 2, f"{calls} places report it"

    def test_the_hint_names_the_source_it_is_recording_from(self):
        js = page_script(page())
        assert 'const from = micLabel ? " from " + micLabel : "";' in js
