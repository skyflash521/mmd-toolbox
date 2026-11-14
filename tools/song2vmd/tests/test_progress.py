import pytest

import vocal_analysis_cli
from song2vmd import progress


@pytest.mark.parametrize("stage_id,label", [
    ("events", "口形イベント確定"),
    ("generate", "モーフ生成"),
])
def test_stage_label_maps_each_tool_specific_stage_id(stage_id, label):
    assert vocal_analysis_cli.stage_label(stage_id, progress.EXTRA_STAGE_LABELS) == label
