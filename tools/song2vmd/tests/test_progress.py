import pytest

import vocal_analysis_cli
from song2vmd import progress


@pytest.mark.parametrize("stage_id,label", [
    ("load", "音声読み込み"),
    ("separate", "ボーカル分離"),
    ("recognize", "音素認識"),
    ("rms", "音量解析"),
    ("events", "口形イベント確定"),
    ("generate", "モーフ生成"),
    ("write", "書き出し"),
])
def test_stage_label_maps_each_stage_id(stage_id, label):
    assert vocal_analysis_cli.stage_label(stage_id, progress.EXTRA_STAGE_LABELS) == label


def test_extra_labels_hold_only_song2vmd_specific_stages():
    assert set(progress.EXTRA_STAGE_LABELS) == {"events", "generate"}
    assert not set(progress.EXTRA_STAGE_LABELS) & set(vocal_analysis_cli.FRONT_STAGE_LABELS)
