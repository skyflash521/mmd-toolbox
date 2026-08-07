"""song2vpr の段ラベル表のテスト。

表の中身だけを検証する。報告先の振り分け(機械/非機械の分岐・同一段の継続・送出失敗の区別)の実体と
その検証は共有の cli_progress_router が持つ。
"""

import pytest

from song2vpr import progress


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
