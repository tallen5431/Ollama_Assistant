"""What a routine records: kinds, and the fields worked out rather than read.

The case behind all of it is real. One Uber trip was logged twice, five minutes
apart. The first row said $26.23 an hour; the second said $23.19 — and the
second had no start or end time in it at all, so its rate had been worked out
from nothing. The model was never asked to divide; it was asked what the answer
said, and it obliged.
"""

from __future__ import annotations

import pytest

import fields
import values


UBER = """
Start odometer: distance
End odometer: distance
Distance traveled = End odometer - Start odometer
Start time: timestamp
End time: timestamp
Elapsed time = End time - Start time
Total earnings: money
Earnings per mile = Total earnings / Distance traveled
Earnings per hour = Total earnings / Elapsed time
Average speed = Distance traveled / Elapsed time
"""

FULL = {
    "Start odometer": "102,072",
    "End odometer": "102,165",
    "Start time": "Tuesday 25 August 2026 at 20:06 (evening) (UTC-04:00)",
    "End time": "Wednesday 26 August 2026 at 00:31 (night) (UTC-04:00)",
    "Total earnings": "$115.94",
}
# The second capture of the same trip: same readings, no times at all.
NO_TIMES = {"Start odometer": "102,072 mi", "End odometer": "102,165 mi",
            "Total earnings": "$115.94"}


class TestReadingADeclaration:
    def test_a_bare_list_of_names_still_means_what_it_did(self):
        """Every routine written before this is a bare list of names."""
        parsed = fields.parse(["distance", "elapsed", "average speed"])
        assert fields.names(parsed) == ["distance", "elapsed", "average speed"]
        assert not any(f.computed or f.kind for f in parsed)

    def test_a_field_can_say_what_it_holds(self):
        parsed = fields.parse(["Total earnings: money"])
        assert parsed[0].name == "Total earnings"
        assert parsed[0].kind == values.MONEY

    def test_and_a_field_can_say_how_it_is_worked_out(self):
        parsed = fields.parse(["Distance = End odo - Start odo"])
        assert parsed[0].computed
        assert (parsed[0].left, parsed[0].op, parsed[0].right) == \
            ("End odo", "-", "Start odo")

    @pytest.mark.parametrize("raw", [
        "a, b, c",
        ["a, b", "c"],
        "a\nb\nc",
        ["a", "b", "c"],
    ])
    def test_lines_or_commas_both_separate(self, raw):
        assert fields.names(fields.parse(raw)) == ["a", "b", "c"]

    def test_an_unknown_kind_is_ignored_rather_than_guessed(self):
        """The column still works; it just infers, as it did before."""
        parsed = fields.parse(["Fare: munny"])
        assert parsed[0].name == "Fare" and parsed[0].kind == ""

    def test_duplicates_are_dropped(self):
        assert len(fields.parse(["Fare: money", "fare"])) == 1

    def test_it_round_trips_through_the_text_it_was_written_as(self):
        parsed = fields.parse(UBER)
        assert fields.parse(fields.declarations(parsed)) == parsed

    def test_nothing_usable_is_nothing(self):
        assert fields.parse("") == []
        assert fields.parse(None) == []
        assert fields.parse(["  ", ""]) == []


class TestOnlyTheReadingsAreAskedFor:
    def test_the_model_is_not_asked_to_do_arithmetic(self):
        parsed = fields.parse(UBER)
        assert len(parsed) == 10
        assert [f.name for f in fields.to_ask(parsed)] == [
            "Start odometer", "End odometer", "Start time", "End time",
            "Total earnings"]

    def test_an_untyped_routine_asks_for_everything(self):
        parsed = fields.parse(["a", "b"])
        assert len(fields.to_ask(parsed)) == 2


class TestWorkingItOut:
    def test_the_full_capture_computes_from_its_own_readings(self):
        got, notes = fields.compute(fields.parse(UBER), FULL)
        assert got["Distance traveled"] == "93 mi"
        assert got["Elapsed time"] == "4h 25m"
        assert got["Earnings per hour"] == "$26.25"
        assert got["Average speed"] == "21.1 mph"
        assert notes == []

    def test_a_rate_with_nothing_to_divide_by_comes_out_empty(self):
        """The whole point. This row is what produced $23.19 an hour."""
        got, notes = fields.compute(fields.parse(UBER), NO_TIMES)
        assert got["Earnings per hour"] == ""
        assert got["Elapsed time"] == ""
        assert any("Elapsed time" in n for n in notes)

    def test_but_what_can_still_be_worked_out_is(self):
        """A missing time must not cost the figures that do not depend on it."""
        got, _ = fields.compute(fields.parse(UBER), NO_TIMES)
        assert got["Distance traveled"] == "93 mi"
        assert got["Earnings per mile"] == "$1.25"

    def test_a_computed_field_can_build_on_another(self):
        """Elapsed time is itself computed, and the hourly rate divides by it."""
        got, _ = fields.compute(fields.parse(UBER), FULL)
        assert got["Elapsed time"] and got["Earnings per hour"]

    def test_the_note_says_which_input_was_missing(self):
        _, notes = fields.compute(fields.parse(UBER), NO_TIMES)
        assert any("End time" in n and "Start time" in n for n in notes)

    def test_dividing_by_zero_is_a_note_not_a_crash(self):
        parsed = fields.parse(["a: money", "b: distance", "c = a / b"])
        got, notes = fields.compute(parsed, {"a": "$10", "b": "0 mi"})
        assert got["c"] == "" and any("zero" in n for n in notes)

    def test_a_date_with_no_time_gives_no_duration(self):
        """Treating a bare date as midnight would produce a confident, wrong
        elapsed time rather than none."""
        parsed = fields.parse(["Start: timestamp", "End: timestamp",
                               "Took = End - Start"])
        got, notes = fields.compute(parsed, {"Start": "12 August 2026",
                                             "End": "13 August 2026"})
        assert got["Took"] == "" and notes

    @pytest.mark.parametrize("left, op, right, expected", [
        (values.DISTANCE, "-", values.DISTANCE, values.DISTANCE),
        (values.TIMESTAMP, "-", values.TIMESTAMP, values.DURATION),
        (values.MONEY, "/", values.DISTANCE, values.MONEY),
        (values.MONEY, "/", values.DURATION, values.MONEY),
        (values.DISTANCE, "/", values.DURATION, values.SPEED),
        (values.MONEY, "-", values.MONEY, values.MONEY),
        (values.SPEED, "*", values.MONEY, values.NUMBER),   # nothing sensible
    ])
    def test_the_result_knows_what_it_is(self, left, op, right, expected):
        assert fields.result_kind(left, op, right) == expected

    def test_a_computed_value_is_rendered_as_its_kind(self):
        assert values.render(values.MONEY, 26.2506) == "$26.25"
        assert values.render(values.SPEED, 21.056) == "21.1 mph"
        assert values.render(values.DURATION, 4.41667) == "4h 25m"
        assert values.render(values.DISTANCE, 93.0) == "93 mi"

    def test_a_negative_result_keeps_its_sign(self):
        assert values.render(values.MONEY, -12.5) == "-$12.50"

    def test_nothing_at_all_renders_as_nothing(self):
        assert values.render(values.MONEY, float("nan")) == ""


class TestADeclarationBeatsAGuess:
    def test_a_declared_kind_wins_over_the_column_vote(self):
        """Inference is a good guess across a column, and a good guess is still
        a guess. Someone writing it down settles it."""
        parsed = fields.parse(["Code: text"])
        rows = [{"Code": "100"}, {"Code": "200"}]        # would infer a number
        assert fields.kinds(parsed, rows)["Code"] == values.TEXT

    def test_an_undeclared_column_is_still_inferred(self):
        parsed = fields.parse(["Odo"])
        rows = [{"Odo": "102,072 mi"}, {"Odo": "102,165"}]
        assert fields.kinds(parsed, rows)["Odo"] == values.DISTANCE


class TestAValueOfTheWrongKindIsFlaggedNotForced:
    """The conservatism the standardiser had to learn the hard way, applied to
    the typed version: a value that does not match its declared kind is
    reported and left exactly as it is."""

    PARSED = fields.parse(["Total earnings: money", "Notes"])

    def test_a_mismatch_is_reported(self):
        wrong = fields.mismatches(self.PARSED, {"Total earnings": "unknown"})
        assert "Total earnings" in wrong
        assert "money" in wrong["Total earnings"]

    def test_a_matching_value_is_not(self):
        assert fields.mismatches(self.PARSED, {"Total earnings": "$40"}) == {}

    def test_an_empty_value_is_a_gap_not_a_mismatch(self):
        assert fields.mismatches(self.PARSED, {"Total earnings": ""}) == {}

    def test_an_undeclared_column_is_never_a_mismatch(self):
        assert fields.mismatches(self.PARSED, {"Notes": "quiet night"}) == {}

    def test_the_value_itself_is_untouched(self):
        """Reported, never corrected. "unknown" where a price was wanted has
        told you something; a blank in its place has not."""
        row = {"Total earnings": "unknown"}
        fields.mismatches(self.PARSED, row)
        assert row["Total earnings"] == "unknown"


class TestTimesComeFromTheFileNotTheModel:
    """The reported failure: even large models get the capture times wrong.

    They are not reading them badly — they are being asked to align two lists
    across two messages by position. The metadata block says "Image 1" and the
    photos carry no labels at all, so the join has no anchor. It is a binding
    problem, not a hard one, which is why model size does not rescue it.

    The app has these times exactly, from the file, before any of that.
    """

    DECL = fields.parse("""
        Start odometer: distance
        End odometer: distance
        Distance = End odometer - Start odometer
        Start time = earliest photo taken
        End time = latest photo taken
        Elapsed time = End time - Start time
        Average speed = Distance / Elapsed time
    """)
    PHOTOS = [{"taken": "2026:08:07 13:37:12", "offset": "-04:00"},
              {"taken": "2026:08:07 16:45:03", "offset": "-04:00"}]

    def test_the_model_is_not_asked_for_them(self):
        asked = [f.name for f in fields.to_ask(self.DECL)]
        assert asked == ["Start odometer", "End odometer"]
        assert "Start time" not in asked

    def test_they_are_read_off_the_file(self):
        got = fields.from_photos(self.DECL, self.PHOTOS)
        assert got["Start time"] == "2026-08-07 13:37 UTC-04:00"
        assert got["End time"] == "2026-08-07 16:45 UTC-04:00"

    def test_the_order_they_were_attached_in_does_not_matter(self):
        """The routine's own body used to warn about this — "do not assume the
        first photo is the start" — because a gallery hands them over in
        whatever order it likes."""
        got = fields.from_photos(self.DECL, list(reversed(self.PHOTOS)))
        assert got["Start time"] == "2026-08-07 13:37 UTC-04:00"
        assert got["End time"] == "2026-08-07 16:45 UTC-04:00"

    def test_a_positional_source_takes_the_photo_it_names(self):
        parsed = fields.parse(["First shot = photo 1 taken",
                               "Second shot = photo 2 taken"])
        got = fields.from_photos(parsed, self.PHOTOS)
        assert got["First shot"] == "2026-08-07 13:37 UTC-04:00"
        assert got["Second shot"] == "2026-08-07 16:45 UTC-04:00"

    def test_the_elapsed_time_built_on_them_is_exact(self):
        row = {"Start odometer": "100,339", "End odometer": "100,407",
               **fields.from_photos(self.DECL, self.PHOTOS)}
        got, notes = fields.compute(self.DECL, row)
        assert got["Elapsed time"] == "3h 08m"
        assert got["Average speed"] == "21.7 mph"
        assert notes == []

    @pytest.mark.parametrize("photos", [None, [], [{}, {}], [{"taken": "rubbish"}]])
    def test_no_readable_exif_leaves_them_empty(self, photos):
        """A screenshot, an edited copy, or the photo-details toggle turned
        off. Empty is the honest answer and the elapsed time goes with it."""
        got = fields.from_photos(self.DECL, photos)
        assert all(v == "" for v in got.values())

    def test_a_photo_index_past_the_end_is_empty_not_an_error(self):
        parsed = fields.parse(["Third = photo 3 taken"])
        assert fields.from_photos(parsed, self.PHOTOS) == {"Third": ""}

    def test_a_source_field_round_trips_through_its_declaration(self):
        assert fields.parse(fields.declarations(self.DECL)) == self.DECL

    def test_a_source_field_is_always_a_timestamp(self):
        field = fields.parse(["When = photo 1 taken"])[0]
        assert field.kind == values.TIMESTAMP
