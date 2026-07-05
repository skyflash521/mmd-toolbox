"""song2vmd CLI 自己記述 --describe のテスト(song2vmd.md 12.2)。

--describe は音声を読まず、オプション定義とプリセット一覧の result イベント(mode:"describe")を
出して終了する独立メタ操作。--machine を要さず単独で起動でき、入力 positional も要求しない。
options は処理を駆動する引数の配列で、各要素は {name, type, constraint, default, help} の5キー
(song2vmd は repeatable な引数を持たないため sparsevmd のような `repeat` キーは持たない)。
真偽フラグの否定形(--no-n-morph)とメタ/モード操作(--describe/--version/--help/--machine)は
options に載せない。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
"""

import json

from song2vmd import cli

_NONNEG_INT = {"min": 0, "max": None, "exclusive_min": False}
_UNIT = {"min": 0, "max": 1, "exclusive_min": False}
_POS = {"min": 0, "max": None, "exclusive_min": True}


def _ffield(name, mn, mx, ex):
    return {"name": name, "type": "float", "min": mn, "max": mx, "exclusive_min": ex}


# song2vmd.md 12.2 の options 全22要素。name → (type, constraint, default)。
EXPECTED = {
    "input": ("str", None, None),
    "--output": ("str", None, None),
    "--overwrite": ("flag", None, False),
    "--model-name": ("str", None, ""),
    "--style": ("enum", {"choices": ["pop", "ballad", "powerful", "whisper", "rap"]}, "pop"),
    "--separate-vocals": ("enum", {"choices": ["auto", "always", "never"]}, "auto"),
    "--separator": ("enum", {"choices": ["audio-separator-htdemucs-ft"]}, "audio-separator-htdemucs-ft"),
    "--recognizer-model-id": ("str", None, None),
    "--recognizer-model-revision": ("str", None, None),
    "--n-morph": ("flag", None, True),
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
    "pop": {"open_max": 0.90, "coarticulation": 2, "anticipation": 1, "min_hold": 3},
    "ballad": {"open_max": 0.70, "coarticulation": 3, "anticipation": 1, "min_hold": 4},
    "powerful": {"open_max": 0.97, "coarticulation": 2, "anticipation": 2, "min_hold": 3},
    "whisper": {"open_max": 0.50, "coarticulation": 2, "anticipation": 1, "min_hold": 3},
    "rap": {"open_max": 0.85, "coarticulation": 1, "anticipation": 1, "min_hold": 2},
}


def describe_result(capsysbinary):
    """--describe の stdout を解析し、単一の result(mode:"describe")イベントを返す。"""
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_stdout_is_valid_json_lines_lf_only(capsysbinary):
    # --describe は --machine 無しでも構造化出力モード。stdout は UTF-8・LF固定(\r 不在)の
    # JSON Lines のみ(song2vmd.md §5.5・12章、CLI インターフェース規約 §4・§10)。
    rc = cli.main(["--describe"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    # --describe は単一の result で終端する(空行混入を許さない。1行1オブジェクトの厳密化)。
    assert len(lines) == 1 and all(lines)
    assert "type" in json.loads(lines[0])


def test_describe_emits_result_without_input(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list) and r["presets"]
    assert set(r) == {"type", "mode", "options", "presets"}


def test_describe_works_without_machine_flag(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_ignores_input_and_does_not_read_audio(capsysbinary):
    """--describe は音声を読まない独立メタ操作。存在しない入力パスを渡してもdescribeで終わる。"""
    rc = cli.main(["--describe", "does_not_exist.wav"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    assert "--no-n-morph" not in by_name
    assert len(r["options"]) == len(EXPECTED)
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert isinstance(o["help"], str) and o["help"]
    assert set(by_name) == set(EXPECTED)
    for name, (type_, constraint, default) in EXPECTED.items():
        o = by_name[name]
        assert o["type"] == type_, name
        assert o["constraint"] == constraint, name
        assert o["default"] == default, name


def test_describe_presets_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    for p in r["presets"]:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {"open_max", "coarticulation", "anticipation", "min_hold"}
    assert {p["name"]: p["values"] for p in r["presets"]} == EXPECTED_PRESETS


def test_describe_type_table_covers_non_meta_args():
    # _D_TYPE はメタ/モード操作を除く全 parser 引数を覆う。parser に引数を足して _D_TYPE への追加を
    # 忘れると describe から黙って抜けるため、その載せ忘れをここで検出する。
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_mode_arg_error_is_error_event(capsysbinary):
    # --describe(--machine 無し)も構造化出力モードなので、引数エラーは標準エラーでなく error
    # イベントでストリームを終端する。
    rc = cli.main(["--describe", "--max-duration", "abc"])
    assert rc == 2
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_argument"
    assert events[-1]["field"] == "--max-duration" and events[-1]["exit_code"] == 2
