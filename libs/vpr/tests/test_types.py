def test_note_is_protected_defaults_to_false():
    from vpr import Note

    note = Note(start_tick=0, duration_tick=1, pitch=60, lyric="あ", velocity=0)
    assert note.is_protected is False


def test_note_is_protected_is_keyword_only_so_the_seventh_positional_binds_vibrato():
    from vpr import Note, NoteVibrato

    vibrato = NoteVibrato(type=0, duration=240)
    note = Note(0, 480, 60, "あ", 64, ["a"], vibrato)
    assert note.vibrato is vibrato
    assert note.is_protected is False
