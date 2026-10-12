import pytest

from song2vpr import progress
from vocal_analysis_cli import stage_label


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
    assert stage_label(stage_id, progress.EXTRA_STAGE_LABELS) == label


def test_stage_label_returns_unknown_id_as_is():
    assert stage_label("未知の工程", progress.EXTRA_STAGE_LABELS) == "未知の工程"


def test_extra_labels_cover_exactly_the_stages_owned_by_the_tool():
    assert set(progress.EXTRA_STAGE_LABELS) == {"f0", "notes"}
