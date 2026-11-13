import pytest

from song2vpr import progress
from vocal_analysis_cli import stage_label


@pytest.mark.parametrize("stage_id,label", [
    ("f0", "ピッチ推定"),
    ("notes", "音符化"),
])
def test_stage_label_maps_each_stage_owned_by_the_tool(stage_id, label):
    assert stage_label(stage_id, progress.EXTRA_STAGE_LABELS) == label
