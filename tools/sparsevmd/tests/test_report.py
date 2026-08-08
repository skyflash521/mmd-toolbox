"""dry-run レポートのテスト。

report は削減の入出力キー数・削減率・選択ボーン・範囲・keep-frame をまとめ、
dry-run のテキスト表示を提供する。
"""

import pytest

from sparsevmd import report


def test_reduction_rate():
    assert report.reduction_rate(100, 25) == pytest.approx(0.75)
    assert report.reduction_rate(0, 0) == 0.0  # 入力0は0扱い(ゼロ除算しない)
    assert report.reduction_rate(10, 10) == pytest.approx(0.0)


def sample_report():
    return report.build_report(
        target="all",
        camera=(31, 2),
        bones={"センター": (31, 3), "頭": (10, 10)},
        selected_bones={"センター"},
        ranges=[(0, 30)],
        keep_frames=[12],
    )


def test_build_report_structure():
    r = sample_report()
    assert r["target"] == "all"
    assert r["camera"]["input_keys"] == 31
    assert r["camera"]["output_keys"] == 2
    assert r["camera"]["reduction_rate"] == pytest.approx(1 - 2 / 31)
    assert len(r["bones"]) == 2  # 全ボーンを列挙
    bones = {b["name"]: b for b in r["bones"]}
    assert bones["センター"]["selected"] is True
    assert bones["センター"]["input_keys"] == 31
    assert bones["センター"]["output_keys"] == 3
    assert bones["頭"]["selected"] is False
    assert bones["頭"]["input_keys"] == 10
    assert bones["頭"]["output_keys"] == 10  # 未選択は保持
    assert r["ranges"] == [(0, 30)]
    assert r["keep_frames"] == [12]


def test_build_report_camera_none():
    r = report.build_report(target="bone", camera=None, bones={"センター": (5, 2)},
                            selected_bones={"センター"}, ranges=[(0, 4)], keep_frames=[])
    assert r["camera"] is None


def test_format_dry_run_contains_counts_rate_selection_range_keep():
    text = report.format_dry_run(sample_report())
    assert "camera" in text
    assert "31" in text and "2" in text  # 入力/出力キー数
    assert "センター" in text and "頭" in text
    assert "%" in text or "0." in text  # 削減率
    # 選択状態・範囲・keep-frame も表示される。
    low = text.lower()
    assert "select" in low or "選択" in text
    assert "12" in text  # keep-frame
    assert "30" in text  # 範囲端


def test_build_report_note_when_not_reducible():
    # 削減対象なし(全トラック1キー以下・範囲空)を記録する。
    rep = report.build_report(
        target="camera", camera=(1, 1), bones=None, selected_bones=set(),
        ranges=[(7, 7)], keep_frames=[], reduced=False,
    )
    assert rep["note"] == "削減対象なし"
    assert "削減対象なし" in report.format_dry_run(rep)


def test_build_report_no_note_when_reducible():
    rep = report.build_report(
        target="camera", camera=(31, 2), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], reduced=True,
    )
    assert "note" not in rep


def test_format_dry_run_handles_camera_none(tmp_path):
    r = report.build_report(
        target="bone", camera=None, bones={"センター": (5, 2)},
        selected_bones={"センター"}, ranges=[(0, 4)], keep_frames=[],
    )
    text = report.format_dry_run(r)  # 例外を出さない
    assert "センター" in text


