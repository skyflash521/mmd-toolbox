import json

from song2vmd import __version__, cli

_NONNEG_INT = {"min": 0, "max": None, "exclusive_min": False}
_UNIT = {"min": 0, "max": 1, "exclusive_min": False}
_POS = {"min": 0, "max": None, "exclusive_min": True}


def _ffield(name, mn, mx, ex):
    return {"name": name, "type": "float", "min": mn, "max": mx, "exclusive_min": ex}


EXPECTED_TYPE_CONSTRAINT_DEFAULT = {
    "input": ("str", None, None),
    "--output": ("str", None, None),
    "--overwrite": ("flag", None, False),
    "--model-name": ("str", None, f"song2vmd {__version__}"),
    "--style": ("enum", {"choices": ["pop", "ballad", "powerful", "whisper", "rap"]}, "pop"),
    "--separate-vocals": ("enum", {"choices": ["always", "never"]}, "always"),
    "--separator": ("enum", {"choices": ["audio-separator-htdemucs-ft"]}, "audio-separator-htdemucs-ft"),
    "--recognizer-model-id": ("str", None, None),
    "--recognizer-model-revision": ("str", None, None),
    "--recognizer-retry": ("flag", None, True),
    "--forced-aligner": ("enum", {"choices": ["wav2vec2-ctc-forcedalign", "sofa-forcedalign"]},
                         "wav2vec2-ctc-forcedalign"),
    "--english-katakana-method": ("enum", {"choices": ["arpakana", "tinyllama-katakana-converter"]},
                                      "arpakana"),
    "--device": ("enum", {"choices": ["auto", "cpu"]}, "auto"),
    "--sofa-python": ("str", None, None),
    "--sofa-root": ("str", None, None),
    "--sofa-checkpoint": ("str", None, None),
    "--sofa-timeout": ("float", _POS, 300.0),
    "--n-morph": ("flag", None, False),
    "--vowel-gain": ("compound",
                     {"format": "a:i:u:e:o",
                      "fields": [_ffield(n, 0, None, False) for n in ("a", "i", "u", "e", "o")]},
                     [1.0, 1.0, 1.0, 1.0, 1.0]),
    "--open-max": ("float", _UNIT, None),
    "--coarticulation": ("int", _NONNEG_INT, None),
    "--anticipation": ("int", _NONNEG_INT, None),
    "--min-hold": ("int", _NONNEG_INT, None),
    "--intensity-curve": ("float", _POS, 0.6),
    "--silence-threshold": ("compound",
                           {"format": "ON:OFF",
                            "fields": [_ffield("ON", 0, 1, False), _ffield("OFF", 0, 1, False)]},
                           [0.06, 0.10]),
    "--max-duration": ("float", {"min": 0, "max": None, "exclusive_min": False}, 300.0),
    "--dry-run": ("flag", None, False),
    "--keep-intermediate": ("flag", None, False),
    "--verbose": ("flag", None, False),
    "--quiet": ("flag", None, False),
}

EXPECTED_PRESETS = {
    "pop": {"open_max": 0.90, "coarticulation": 6, "anticipation": 11, "min_hold": 1},
    "ballad": {"open_max": 0.70, "coarticulation": 3, "anticipation": 1, "min_hold": 4},
    "powerful": {"open_max": 0.97, "coarticulation": 2, "anticipation": 2, "min_hold": 3},
    "whisper": {"open_max": 0.50, "coarticulation": 2, "anticipation": 1, "min_hold": 3},
    "rap": {"open_max": 0.85, "coarticulation": 1, "anticipation": 1, "min_hold": 2},
}


def single_describe_result(capsysbinary):
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_stdout_is_a_single_lf_terminated_json_line(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 1 and all(lines)
    assert "type" in json.loads(lines[0])


def test_describe_emits_result_without_input(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list) and r["presets"]
    assert set(r) == {"type", "mode", "options", "presets"}


def test_describe_with_nonexistent_input_still_describes(capsysbinary):
    rc = cli.main(["--describe", "does_not_exist.wav"])
    assert rc == 0
    assert single_describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    assert "--no-n-morph" not in by_name
    assert len(r["options"]) == len(EXPECTED_TYPE_CONSTRAINT_DEFAULT)
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help", "repeat"}
        assert isinstance(o["help"], str) and o["help"]
    assert set(by_name) == set(EXPECTED_TYPE_CONSTRAINT_DEFAULT)
    for name, (type_, constraint, default) in EXPECTED_TYPE_CONSTRAINT_DEFAULT.items():
        o = by_name[name]
        assert o["type"] == type_, name
        assert o["constraint"] == constraint, name
        assert o["default"] == default, name


def test_describe_presets_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = single_describe_result(capsysbinary)
    for p in r["presets"]:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {"open_max", "coarticulation", "anticipation", "min_hold"}
    assert {p["name"]: p["values"] for p in r["presets"]} == EXPECTED_PRESETS


def test_describe_without_machine_reports_arg_error_as_error_event(capsysbinary):
    rc = cli.main(["--describe", "--max-duration", "abc"])
    assert rc == 2
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_argument"
    assert events[-1]["field"] == "--max-duration" and events[-1]["exit_code"] == 2

