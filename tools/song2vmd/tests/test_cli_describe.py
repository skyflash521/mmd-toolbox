"""song2vmd CLI 自己記述 --describe のテスト。

--describe は音声を読まず、オプション定義とプリセット一覧の result イベント(mode:"describe")を
出して終了する独立メタ操作。--machine を要さず単独で起動でき、入力 positional も要求しない。
options は処理を駆動する引数の配列で、各要素は {name, type, constraint, default, help} の5キー
(song2vmd は repeatable な引数を持たないため sparsevmd のような `repeat` キーは持たない)。
真偽フラグの否定形(--no-n-morph)とメタ/モード操作(--describe/--version/--help/--machine)は
options に載せない。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
"""

import argparse
import json
import math

import pytest

from cli_options import CompoundValidator, RangeValidator, describe_options
from song2vmd import __version__, cli

_NONNEG_INT = {"min": 0, "max": None, "exclusive_min": False}
_UNIT = {"min": 0, "max": 1, "exclusive_min": False}
_POS = {"min": 0, "max": None, "exclusive_min": True}


def _ffield(name, mn, mx, ex):
    return {"name": name, "type": "float", "min": mn, "max": mx, "exclusive_min": ex}


# --describe が返す options 全30要素。name → (type, constraint, default)。
EXPECTED = {
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


def describe_result(capsysbinary):
    """--describe の stdout を解析し、単一の result(mode:"describe")イベントを返す。"""
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_stdout_is_valid_json_lines_lf_only(capsysbinary):
    # --describe は --machine 無しでも構造化出力モード。stdout は UTF-8・LF固定(\r 不在)の
    # JSON Lines のみ。
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

# --- constraint と引数検証の一致 ------------------------------------------------


def _actions_by_describe_name(parser):
    """--describe の name(長形式フラグ / input)から parser の action を引く辞書。"""
    actions = {}
    for action in parser._actions:
        if action.dest == "input":
            actions["input"] = action
            continue
        positive = [s for s in action.option_strings
                    if s.startswith("--") and not s.startswith("--no-")]
        if positive:
            actions[positive[0]] = action
    return actions


def _options_by_name(parser):
    return {o["name"]: o for o in describe_options(parser, cli._D_TYPE)}


def _rejects(validator, text):
    """検証子が text を引数エラーとして弾くか。"""
    try:
        validator(text)
    except (argparse.ArgumentTypeError, ValueError):
        return True
    return False


_NUMERIC_NAMES = [n for n, (type_, c, _) in EXPECTED.items()
                  if type_ in ("int", "float") and c is not None]
_COMPOUND_NAMES = [n for n, (type_, _, _) in EXPECTED.items() if type_ == "compound"]

# 複合値は要素1つだけを範囲外にした値を作る土台が要る(他の要素は受理される値にする)。
# --silence-threshold は要素間の関係の制約があるため、土台もそれを満たす値にする。
_COMPOUND_BASELINES = {
    "--vowel-gain": [1.0, 1.0, 1.0, 1.0, 1.0],
    "--silence-threshold": [0.1, 0.5],
}

# 境界そのものが受理されることを複合値の経路で見られる (要素, 境界) の組。土台と組んだときに
# 要素間の関係の制約へ抵触する組(--silence-threshold の ON の上限・OFF の下限)は、要素単体を
# 直接呼ぶ導出テストで見るのでここには含めない。
_COMPOUND_ACCEPTED = {
    "--vowel-gain": [(i, "min") for i in range(5)],
    "--silence-threshold": [(0, "min"), (1, "max")],
}


@pytest.mark.parametrize("name", _COMPOUND_NAMES)
def test_compound_field_constraint_agrees_with_argument_validation(name):
    # 複合値も要素ごとに範囲を公開するので、要素単位で受理範囲と一致することを確かめる。
    parser = cli._build_parser()
    fields = _options_by_name(parser)[name]["constraint"]["fields"]
    validator = _actions_by_describe_name(parser)[name].type

    def value_with(index, element):
        values = list(_COMPOUND_BASELINES[name])
        values[index] = element
        return ":".join(repr(float(v)) for v in values)

    for index, field in enumerate(fields):
        minimum, maximum = field["min"], field["max"]
        if minimum is not None:
            if field["exclusive_min"]:
                assert _rejects(validator, value_with(index, minimum))
            assert _rejects(validator, value_with(index, math.nextafter(float(minimum), -math.inf)))
        if maximum is not None:
            assert _rejects(validator, value_with(index, math.nextafter(float(maximum), math.inf)))
    for index, bound in _COMPOUND_ACCEPTED[name]:
        assert not _rejects(validator, value_with(index, fields[index][bound]))


@pytest.mark.parametrize("name", _NUMERIC_NAMES)
def test_numeric_constraint_is_derived_from_the_validator(name):
    # 手書きの複製だと検証側だけを直したときに黙って食い違うので、同じ範囲から両方を導く。
    parser = cli._build_parser()
    validator = _actions_by_describe_name(parser)[name].type
    assert isinstance(validator, RangeValidator)
    assert _options_by_name(parser)[name]["constraint"] == validator.constraint


@pytest.mark.parametrize("name", _COMPOUND_NAMES)
def test_compound_constraint_is_derived_from_the_element_validators(name):
    # 複合値も、書式と要素の範囲を検証子自身から取る(要素の検証も同じ範囲で行う)。
    parser = cli._build_parser()
    validator = _actions_by_describe_name(parser)[name].type
    assert isinstance(validator, CompoundValidator)

    constraint = _options_by_name(parser)[name]["constraint"]
    assert constraint["format"] == validator.format
    assert [f["name"] for f in constraint["fields"]] == [n for n, _ in validator.elements]
    for field, (_, element) in zip(constraint["fields"], validator.elements, strict=True):
        assert isinstance(element, RangeValidator)
        assert {k: field[k] for k in ("min", "max", "exclusive_min")} == element.constraint
