"""song2vpr CLI 自己記述 --describe のテスト。

--describe は音声を読まず、オプション定義とプリセット一覧の result イベント(mode:"describe")を
出して終了する独立メタ操作。--machine を要さず単独で起動でき、入力 positional も要求しない。
options は処理を駆動する引数の配列で、各要素は {name, type, constraint, default, help} の5キー。
メタ/モード操作(--describe/--version/--help/--machine)と真偽フラグの否定形は options に載せない。
song2vpr はプリセットを持たないため presets は常に空配列。

機械モード標準出力は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
"""

import json

import pytest

from cli_options import RangeValidator

cli = pytest.importorskip("song2vpr.cli", reason="impl pending: T-1 パッケージ雛形とCLI骨組み")

# メタ/モード操作は options に載せない。
_EXCLUDED = {"--describe", "--version", "--help", "--machine"}

# 音声前段の共通引数群(全件公開する)。名前・意味・既定は共有側が定めるので、ここでは
# 載っていることだけを見る(値まで固定すると共有側の変更へ追随できなくなる)。
_SHARED_OPTIONS = {
    "--separate-vocals", "--separator", "--recognizer-model-id", "--recognizer-model-revision",
    "--recognizer-retry", "--forced-aligner", "--sofa-python", "--sofa-root", "--sofa-checkpoint",
    "--sofa-timeout", "--english-katakana-method", "--device", "--max-duration",
}

# song2vpr 自身が持つ引数の (type, constraint, default)。値の正本は要求仕様書 §9.2。
_OWN_OPTIONS = {
    "input": ("str", None, None),
    "--output": ("str", None, None),
    "--overwrite": ("flag", None, False),
    "--dry-run": ("flag", None, False),
    "--keep-intermediate": ("flag", None, False),
    "--verbose": ("flag", None, False),
    "--quiet": ("flag", None, False),
}

# 制約を検証子から導く数値引数(共通引数群のうち範囲を持つもの)。
_NUMERIC_NAMES = ["--sofa-timeout", "--max-duration"]


def _describe_raw(capsysbinary, argv=("--describe",)):
    assert cli.main(list(argv)) == 0
    return capsysbinary.readouterr().out


def _describe(capsysbinary, argv=("--describe",)):
    text = _describe_raw(capsysbinary, argv).decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert len(events) == 1
    return events[0]


def _options_by_name(capsysbinary):
    return {option["name"]: option for option in _describe(capsysbinary)["options"]}


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


# --- 起動と終端 --------------------------------------------------------------


def test_describe_runs_without_input_and_terminates(capsysbinary):
    """入力 positional 無しで起動でき、result ちょうど1つで終端する。"""
    event = _describe(capsysbinary)
    assert event["type"] == "result"
    assert event["mode"] == "describe"


def test_describe_result_has_exactly_four_keys(capsysbinary):
    """mode:"describe" は処理統計のキーを載せない。"""
    event = _describe(capsysbinary)
    assert set(event) == {"type", "mode", "options", "presets"}


def test_describe_does_not_require_machine_flag(capsysbinary):
    """--machine を要さず単独で成立する(標準出力はイベント専用)。"""
    event = _describe(capsysbinary)
    assert "options" in event and "presets" in event


def test_describe_ignores_input_path(tmp_path, capsysbinary):
    """入力パスを渡しても音声を読まず describe で終わる(存在しないパスでも成立する)。"""
    event = _describe(capsysbinary, ("--describe", str(tmp_path / "absent.wav")))
    assert event["mode"] == "describe"


def test_describe_stdout_is_lf_terminated_json_lines(capsysbinary):
    """標準出力は LF 区切りの JSON Lines(CR を混ぜず、空行も入れない)。"""
    raw = _describe_raw(capsysbinary)
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").split("\n")[:-1]
    assert lines and all(line for line in lines)
    for line in lines:
        json.loads(line)


# --- options の掲載対象 ------------------------------------------------------


def test_options_have_five_keys_each(capsysbinary):
    for option in _options_by_name(capsysbinary).values():
        assert set(option) == {"name", "type", "constraint", "default", "help"}


def test_options_are_exactly_own_and_shared_arguments(capsysbinary):
    """掲載対象は固有引数と共通引数群のちょうど全部(余分も欠落も重複も無い)。"""
    expected = set(_OWN_OPTIONS) | _SHARED_OPTIONS
    options = _describe(capsysbinary)["options"]
    assert len(options) == len(expected)  # 同名の要素が2つ載ることも防ぐ
    assert {option["name"] for option in options} == expected


def test_options_exclude_meta_and_mode_operations(capsysbinary):
    assert not (set(_options_by_name(capsysbinary)) & _EXCLUDED)


def test_options_exclude_negated_boolean_flag(capsysbinary):
    """対の真偽フラグは肯定形だけを載せる(--x/--no-x の様式)。"""
    assert "--no-recognizer-retry" not in _options_by_name(capsysbinary)


def test_option_help_is_non_empty(capsysbinary):
    for option in _options_by_name(capsysbinary).values():
        assert option["help"]


# --- options の値 ------------------------------------------------------------


@pytest.mark.parametrize("name, expected", sorted(_OWN_OPTIONS.items()))
def test_own_option_type_constraint_default(capsysbinary, name, expected):
    option = _options_by_name(capsysbinary)[name]
    assert (option["type"], option["constraint"], option["default"]) == expected


@pytest.mark.parametrize("name", _NUMERIC_NAMES)
def test_numeric_constraint_is_derived_from_the_validator(capsysbinary, name):
    # 手書きの複製だと検証側だけを直したときに黙って食い違うので、同じ範囲から両方を導く。
    parser = cli._build_parser()
    validator = _actions_by_describe_name(parser)[name].type
    assert isinstance(validator, RangeValidator)
    assert _options_by_name(capsysbinary)[name]["constraint"] == validator.constraint


def test_presets_is_empty_list(capsysbinary):
    """song2vpr はプリセットを持たない。"""
    assert _describe(capsysbinary)["presets"] == []


# --- 型表の網羅と構造化出力モードとしての振る舞い ----------------------------


def test_type_table_covers_non_meta_args():
    # 型表はメタ/モード操作を除く全 parser 引数を覆う。parser に引数を足して型表への追加を忘れると
    # describe から黙って抜けるため、その載せ忘れをここで検出する。
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_mode_arg_error_is_error_event(capsysbinary):
    # --describe(--machine 無し)も構造化出力モードなので、引数エラーは標準エラーでなく error
    # イベントでストリームを終端する。
    assert cli.main(["--describe", "--max-duration", "abc"]) == 2
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_argument"
    assert events[-1]["field"] == "--max-duration"
    assert events[-1]["exit_code"] == 2
