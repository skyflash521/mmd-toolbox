def _note(start, duration):
    from vpr import Note

    return Note(start_tick=start, duration_tick=duration, pitch=60, lyric="x", velocity=64)


def test_rest_intervals_no_notes_is_all_silence():
    from vpr import rest_intervals

    assert rest_intervals([], 1920) == [(0, 1920)]


def test_rest_intervals_single_note_yields_leading_and_trailing():
    from vpr import rest_intervals

    assert rest_intervals([_note(480, 240)], 1920) == [(0, 480), (720, 1920)]


def test_rest_intervals_note_from_zero_yields_only_trailing():
    from vpr import rest_intervals

    assert rest_intervals([_note(0, 480)], 960) == [(480, 960)]


def test_rest_intervals_note_filling_range_yields_no_rest():
    from vpr import rest_intervals

    assert rest_intervals([_note(0, 1920)], 1920) == []


def test_rest_intervals_adjacent_notes_make_no_false_rest():
    from vpr import rest_intervals

    notes = [_note(480, 240), _note(720, 240)]
    assert rest_intervals(notes, 1920) == [(0, 480), (960, 1920)]


def test_rest_intervals_overlapping_notes_make_no_false_rest():
    from vpr import rest_intervals

    notes = [_note(480, 240), _note(600, 240)]
    assert rest_intervals(notes, 1920) == [(0, 480), (840, 1920)]


def test_rest_intervals_unsorted_input():
    from vpr import rest_intervals

    notes = [_note(960, 240), _note(480, 240)]
    assert rest_intervals(notes, 1920) == [(0, 480), (720, 960), (1200, 1920)]
