import re

from vmd import read as vmd_read
from vpr import (
    ControllerCurve,
    ControllerEvent,
    Note,
    NoteVibrato,
    Part,
    TempoEvent,
    Track,
    VibratoPoint,
    VprFormatError,
    VprProject,
)
from vpr2vmd import __version__, cli


def _note(start, dur, phonemes, *, velocity=64):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x",
        velocity=velocity, phonemes=phonemes,
    )


def _project(notes, *, tracks=None, tempos=None, resolution=480):
    part = Part(name="part", start_tick=0, notes=notes)
    track = Track(name="Vocal", parts=[part])
    return VprProject(
        resolution=resolution,
        tempos=tempos if tempos is not None else [TempoEvent(0, 120.0)],
        tracks=tracks if tracks is not None else [track],
    )


def _patch_read(monkeypatch, project):
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))


def _run(monkeypatch, tmp_path, project, *args):
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out = tmp_path / "out.vmd"
    _patch_read(monkeypatch, project)
    rc = cli.main([str(src), "-o", str(out), *args])
    return rc, out


def _read_doc(path):
    doc, _ = vmd_read(str(path))
    return doc


def _morph_names(path):
    return [k.name for k in _read_doc(path).morph]


def test_convert_single_vowel_writes_morph_vmd(monkeypatch, tmp_path):
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]))
    assert rc == 0
    assert out.exists()
    assert "あ" in _morph_names(out)


def test_convert_default_model_name_is_tool_and_version(monkeypatch, tmp_path):
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]))
    assert rc == 0
    assert _read_doc(out).model_name == f"vpr2vmd {__version__}"


def test_convert_output_has_frame0_keys_for_used_morphs(monkeypatch, tmp_path):
    rc, out = _run(
        monkeypatch, tmp_path, _project([_note(0, 240, ["a"]), _note(480, 240, ["i"])])
    )
    assert rc == 0
    doc = _read_doc(out)
    used = {k.name for k in doc.morph}
    zero = {k.name for k in doc.morph if k.frame == 0}
    assert used
    assert used <= zero


def test_convert_cli_overrides_reach_generation_params(monkeypatch, tmp_path):
    captured = {}

    real = cli.generate_morph_keys

    def spy(events, params):
        captured["params"] = params
        return real(events, params)

    monkeypatch.setattr(cli, "generate_morph_keys", spy)
    rc, _out = _run(
        monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]),
        "--anticipation", "7", "--coartic-overlap", "5",
        "--valley-shallow", "0.5", "--valley-deep", "0.25", "--valley-slope", "0.03",
    )
    assert rc == 0
    p = captured["params"]
    assert p.anticipation_frames == 7
    assert p.coartic_overlap_max == 5
    assert (p.legato_valley_shallow, p.legato_valley_deep, p.legato_valley_slope) == (
        0.5, 0.25, 0.03,
    )


def test_convert_ref_bpm_equal_to_song_tempo_keeps_preset_min_hold(monkeypatch, tmp_path):
    captured = {}

    real = cli.generate_morph_keys

    def spy(events, params):
        captured["params"] = params
        return real(events, params)

    monkeypatch.setattr(cli, "generate_morph_keys", spy)
    project = _project([_note(0, 480, ["a"])], tempos=[TempoEvent(0, 190.0)])
    rc, _out = _run(monkeypatch, tmp_path, project, "--ref-bpm", "190")
    assert rc == 0
    assert captured["params"].min_hold_frames == 3


def test_convert_loudness_controller_drives_open_amount(monkeypatch, tmp_path):
    captured = {}

    real = cli.build_mouth_events

    def spy(*args, **kwargs):
        captured["open"] = kwargs.get("open_by_note")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "build_mouth_events", spy)
    part = Part(
        name="p",
        start_tick=0,
        notes=[_note(0, 240, ["a"]), _note(480, 240, ["a"])],
        controllers=[
            ControllerCurve(
                name="dynamics",
                events=[ControllerEvent(0, 120), ControllerEvent(480, 10)],
            )
        ],
    )
    project = VprProject(
        resolution=480, tempos=[TempoEvent(0, 120.0)], tracks=[Track(name="Vocal", parts=[part])]
    )
    rc, _out = _run(monkeypatch, tmp_path, project)
    assert rc == 0
    assert captured["open"][0] > captured["open"][1]


def test_convert_tempo_scale_min_override_reaches_correction(monkeypatch, tmp_path):
    captured = {}

    real = cli.generate_morph_keys

    def spy(events, params):
        captured["params"] = params
        return real(events, params)

    monkeypatch.setattr(cli, "generate_morph_keys", spy)
    project = _project([_note(0, 480, ["a"])], tempos=[TempoEvent(0, 600.0)])
    rc, _out = _run(monkeypatch, tmp_path, project, "--tempo-scale-min", "0.2")
    assert rc == 0
    assert captured["params"].min_hold_frames == 1


def test_convert_valley_deep_above_shallow_is_arg_error(monkeypatch, tmp_path):
    rc, _out = _run(
        monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--valley-deep", "0.6"
    )
    assert rc == 2


def test_convert_legato_max_override_reaches_build_mouth_events(monkeypatch, tmp_path):
    captured = {}

    real = cli.build_mouth_events

    def spy(*args, **kwargs):
        captured["legato"] = kwargs.get("legato_max_frames")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "build_mouth_events", spy)
    rc, _out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--legato-max", "12")
    assert rc == 0
    assert captured["legato"] == 12.0


def test_convert_writes_only_morph_section(monkeypatch, tmp_path):
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]))
    assert rc == 0
    doc = _read_doc(out)
    assert doc.morph
    assert doc.bone == []
    assert doc.camera == []
    assert doc.light == []
    assert doc.self_shadow == []
    assert doc.ik_property == []


def test_convert_empty_notes_writes_empty_morph_vmd(monkeypatch, tmp_path):
    rc, out = _run(monkeypatch, tmp_path, _project([]))
    assert rc == 0
    assert out.exists()
    assert _morph_names(out) == []


def test_convert_no_tracks_is_input_error(monkeypatch, tmp_path):
    rc, _ = _run(monkeypatch, tmp_path, _project([], tracks=[]))
    assert rc == 1


def test_convert_track_index_out_of_range_is_arg_error(monkeypatch, tmp_path):
    rc, _ = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--track", "5")
    assert rc == 2


def test_convert_track_name_no_match_is_arg_error(monkeypatch, tmp_path):
    rc, _ = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--track", "Nope")
    assert rc == 2


def test_convert_vpr_format_error_is_input_error(monkeypatch, tmp_path):
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out = tmp_path / "out.vmd"

    def _raise(_src):
        raise VprFormatError("not a vpr")

    monkeypatch.setattr(cli, "read", _raise)
    assert cli.main([str(src), "-o", str(out)]) == 1


def test_convert_moraic_nasal_is_silence_by_default(monkeypatch, tmp_path):
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["N\\"])]))
    assert rc == 0
    assert _morph_names(out) == []


def test_convert_no_n_morph_drops_n_morph(monkeypatch, tmp_path):
    rc, out = _run(
        monkeypatch, tmp_path, _project([_note(0, 480, ["N\\"])]), "--no-n-morph"
    )
    assert rc == 0
    assert _morph_names(out) == []


def test_convert_n_morph_flag_enables_n_morph(monkeypatch, tmp_path):
    rc, out = _run(
        monkeypatch, tmp_path, _project([_note(0, 480, ["N\\"])]), "--n-morph"
    )
    assert rc == 0
    assert "ん" in _morph_names(out)


def test_convert_write_failure_is_output_error(monkeypatch, tmp_path):
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out = tmp_path / "out.vmd"
    _patch_read(monkeypatch, _project([_note(0, 480, ["a"])]))

    def _raise(_doc, _path):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_file", _raise)
    assert cli.main([str(src), "-o", str(out)]) == 3


def _dry_run(monkeypatch, tmp_path, project, capsys, *args):
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    _patch_read(monkeypatch, project)
    rc = cli.main([str(src), "--dry-run", *args])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_dry_run_reports_adopted_event_and_morph_counts(monkeypatch, tmp_path, capsys):
    project = _project([_note(0, 240, ["a"]), _note(480, 240, ["i"])])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "採用音符数: 2" in out
    ev = re.search(r"口形イベント数:\s*(\d+)", out)
    assert ev and int(ev.group(1)) >= 1
    mk = re.search(r"モーフキー数:\s*(\d+)", out)
    assert mk and int(mk.group(1)) >= 1


def test_dry_run_reports_openness_stats(monkeypatch, tmp_path, capsys):
    project = _project(
        [_note(0, 240, ["a"], velocity=40), _note(480, 240, ["i"], velocity=120)]
    )
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    m = re.search(r"開き量[^\n]*?([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", out)
    assert m, "開き量の最小/最大/平均が出ていない"
    lo, hi, avg = (float(m.group(i)) for i in (1, 2, 3))
    assert lo < hi
    assert lo <= avg <= hi


def test_dry_run_reports_vowel_undetermined_count(monkeypatch, tmp_path, capsys):
    project = _project([_note(0, 240, ["a"]), _note(480, 240, ["k"])])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "母音未確定: 1" in out


def test_dry_run_reports_overlap_exclusion_and_truncation(monkeypatch, tmp_path, capsys):
    project = _project([
        _note(0, 480, ["a"]),
        _note(0, 240, ["i"]),
        _note(480, 480, ["u"]),
        _note(720, 240, ["e"]),
    ])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "除外: 1" in out
    assert "切り詰め: 1" in out


def test_dry_run_lists_non_event_symbols(monkeypatch, tmp_path, capsys):
    project = _project([_note(0, 480, ["k", "a"])])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "イベント外記号" in out
    assert "k(1)" in out


def test_dry_run_empty_track_warns_on_stderr(monkeypatch, tmp_path, capsys):
    rc, _out, err = _dry_run(monkeypatch, tmp_path, _project([]), capsys)
    assert rc == 0
    assert "warning: no_adopted_notes: " in err


def test_empty_track_warns_on_stderr_in_normal_run(monkeypatch, tmp_path, capsys):
    rc, out = _run(monkeypatch, tmp_path, _project([]))
    assert rc == 0
    assert out.exists()
    assert "warning: no_adopted_notes: " in capsys.readouterr().err


def test_dry_run_non_event_symbols_keep_length_mark(monkeypatch, tmp_path, capsys):
    project = _project([_note(0, 480, ["k:", "a"])])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "k:(1)" in out


def test_pitch_and_vibrato_do_not_change_output(monkeypatch, tmp_path):
    plain = [_note(0, 480, ["a"]), _note(480, 960, ["o"])]
    expressive = [
        Note(start_tick=0, duration_tick=480, pitch=48, lyric="x", velocity=64, phonemes=["a"]),
        Note(
            start_tick=480, duration_tick=960, pitch=79, lyric="x", velocity=64, phonemes=["o"],
            vibrato=NoteVibrato(
                type=1, duration=960,
                depths=[VibratoPoint(pos=480, value=100)], rates=[VibratoPoint(pos=480, value=100)],
            ),
        ),
    ]
    plain_dir = tmp_path / "plain"
    expressive_dir = tmp_path / "expressive"
    plain_dir.mkdir()
    expressive_dir.mkdir()
    _, plain_out = _run(monkeypatch, plain_dir, _project(plain))
    _, expressive_out = _run(monkeypatch, expressive_dir, _project(expressive))
    assert expressive_out.read_bytes() == plain_out.read_bytes()
