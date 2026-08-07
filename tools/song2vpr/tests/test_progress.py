"""song2vpr の段ラベル表と、進捗の振り分けの構築設定のテスト。

振り分けそのもの(機械/非機械の分岐・同一段の継続・送出失敗の区別)の実体と検証は共有の
cli_progress_router が持つ。ここでは song2vpr が渡す設定——段ラベル表と標準エラーの排他ロック——と、
ラベル表の中身だけを検証する。
"""

import io

import pytest

from song2vpr import progress
from vocal_analysis.quiet import STDERR_WRITE_LOCK


@pytest.mark.parametrize("stage_id,label", [
    ("load", "音声読み込み"),
    ("separate", "ボーカル分離"),
    ("recognize", "音素認識"),
    ("rms", "音量解析"),
    ("f0", "ピッチ推定"),
    ("notes", "音符化"),
    ("write", "書き出し"),
])
def test_stage_label_maps_each_stage_id(stage_id, label):
    assert progress.stage_label(stage_id) == label


def test_stage_label_returns_unknown_id_as_is():
    assert progress.stage_label("未知の工程") == "未知の工程"


def test_label_table_covers_exactly_the_documented_stages():
    # 段を足してラベルを足し忘れると、利用者向けの表示に生の id が出る。
    assert set(progress.STAGE_LABELS) == {
        "load", "separate", "recognize", "rms", "f0", "notes", "write"}


@pytest.mark.parametrize("machine,quiet", [(True, False), (False, True), (False, False)])
def test_build_router_passes_labels_and_stderr_write_lock(monkeypatch, machine, quiet):
    # 音声前段は処理中に標準エラーの出力先を一時的に差し替えるので、同じ排他を進捗表示へ渡さないと
    # ライブ行の書き込みが失敗しうる。
    captured = {}

    def fake_router(**kwargs):
        captured.update(kwargs)
        return "router"

    monkeypatch.setattr(progress, "ProgressRouter", fake_router)
    stream = io.StringIO()
    emitter = object()
    assert progress.build_router(
        machine=machine, quiet=quiet, emitter=emitter, stream=stream) == "router"
    assert captured == {
        "machine": machine, "quiet": quiet, "emitter": emitter, "stream": stream,
        "labels": progress.STAGE_LABELS, "write_lock": STDERR_WRITE_LOCK,
    }
