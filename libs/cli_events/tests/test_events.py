"""cli_events の単体テスト(cli_events.md §5)。

機械モードのイベント送出基盤(JSON Lines エミッタ・終端規則・UTF-8・argparse エラー変換)を検証する。
テスト方針は ../../vmd/vmd.md §4 に準ずる(決定論的・外部依存・ネットワークなし)。
"""
import io
import json

import pytest

try:
    from cli_events import (
        EVENT_TYPES,
        ArgumentParseError,
        EventEmitter,
        MachineArgumentParser,
        StreamTerminatedError,
        argparse_error_event,
        error_event,
    )

    _IMPORTED = True
except ImportError:
    _IMPORTED = False

pytestmark = pytest.mark.skipif(not _IMPORTED, reason="impl pending: Step1 cli_events")


def _lines(buf):
    """BytesIO の中身を UTF-8 で行配列(末尾改行除く)に分解する。"""
    return [ln for ln in buf.getvalue().decode("utf-8").split("\n") if ln]


def test_event_type_vocabulary():
    # 規約 §4 の 4 種をこの順で公開する。
    assert EVENT_TYPES == ("progress", "warning", "result", "error")


def test_emits_json_lines_with_type():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.progress(stage="bake", done=0, total=None)
    em.result(mode="bake", output="x.vmd")
    raw = buf.getvalue().decode("utf-8")
    # 末尾改行で終わり、オブジェクト間に空行を挿入しない(1行1オブジェクト、cli_events.md §3)。
    assert raw.endswith("\n")
    parts = raw.split("\n")
    assert parts[-1] == ""  # 末尾改行の後ろは空のみ
    obj_lines = parts[:-1]
    assert len(obj_lines) == 2  # ちょうど2行
    assert all(ln != "" for ln in obj_lines)  # 間に空行が無い(空行を明示拒否)
    objs = [json.loads(ln) for ln in obj_lines]  # 各行が単一 JSON としてパースできる
    assert objs[0]["type"] == "progress"
    assert objs[0]["stage"] == "bake"
    assert objs[0]["total"] is None
    assert objs[1]["type"] == "result"
    assert objs[1]["mode"] == "bake"


def test_progress_and_warning_are_non_terminal():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.progress(stage="bake", done=1, total=10)
    em.warning(code="fade_shortened", message="...", section=None)
    em.progress(stage="bake", done=2, total=10)
    assert not em.terminated
    assert len(_lines(buf)) == 3


def test_result_terminates_stream():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.result(mode="bake")
    assert em.terminated
    with pytest.raises(StreamTerminatedError):
        em.warning(code="x", message="y", section=None)


def test_error_terminates_stream():
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.error(code="not_vmd", exit_code=1, field="input", path=None, message="bad")
    assert em.terminated
    with pytest.raises(StreamTerminatedError):
        em.result(mode="bake")


def test_exactly_one_terminal_event():
    # result/error は 1 ストリームにつきちょうど 1 つ。2 つ目の終端も拒否する。
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.error(code="x", exit_code=1, field=None, path=None, message="m")
    with pytest.raises(StreamTerminatedError):
        em.error(code="y", exit_code=1, field=None, path=None, message="n")


def test_utf8_not_locale_encoding():
    # 日本語(ロケール cp932 でも UTF-8 で出る)。ensure_ascii=False の実バイト。
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.warning(code="non_camera_sections_passthrough", message="カメラ以外", section=["bone"])
    raw = buf.getvalue()
    assert "カメラ以外".encode("utf-8") in raw
    obj = json.loads(raw.decode("utf-8"))
    assert obj["message"] == "カメラ以外"
    assert obj["section"] == ["bone"]


def test_no_raw_newline_in_line():
    # message 内の改行は JSON エスケープされ、出力は 1 行(末尾改行のみ)。
    buf = io.BytesIO()
    em = EventEmitter(buf)
    em.warning(code="x", message="line1\nline2", section=None)
    text = buf.getvalue().decode("utf-8")
    assert text.count("\n") == 1
    assert json.loads(text)["message"] == "line1\nline2"


def test_error_event_builder():
    ev = error_event(code="bad_argument", message="unknown", exit_code=2, field="--foo")
    assert ev == {
        "type": "error",
        "code": "bad_argument",
        "exit_code": 2,
        "field": "--foo",
        "path": None,
        "message": "unknown",
    }


def test_machine_parser_raises_instead_of_exit():
    p = MachineArgumentParser(prog="x")
    p.add_argument("--n", type=int)
    # 正常解析は通常どおり。
    assert p.parse_args(["--n", "3"]).n == 3
    # エラーは SystemExit でなく ArgumentParseError(message を持つ)。
    with pytest.raises(ArgumentParseError) as exc:
        p.parse_args(["--n", "notint"])
    assert exc.value.message


def test_argparse_error_converts_to_error_event():
    # argparse エラー(=引数エラー)を error イベントへ変換する end-to-end フロー(cli_events.md §4)。
    p = MachineArgumentParser(prog="x")
    p.add_argument("--n", type=int)
    with pytest.raises(ArgumentParseError) as exc:
        p.parse_args(["--n", "notint"])
    ev = argparse_error_event(exc.value, code="bad_argument", field="--n")
    assert ev["type"] == "error"
    assert ev["code"] == "bad_argument"
    assert ev["exit_code"] == 2  # argparse エラーは基底共通 2(規約 §5、ヘルパが固定)
    assert ev["field"] == "--n"
    assert ev["path"] is None
    assert ev["message"]  # argparse のメッセージを載せる
