"""レポート(dry-run統計・JSON・CSV)のテスト(sparsevmd.md §2.7)。

report は削減の入出力キー数・削減率・選択ボーン・範囲・keep-frame をまとめ、
dry-run のテキスト表示、JSON 出力、CSV プレビューを提供する。
"""

import csv
import json

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
    # 選択状態・範囲・keep-frame も表示される(§2.7)。
    low = text.lower()
    assert "select" in low or "選択" in text
    assert "12" in text  # keep-frame
    assert "30" in text  # 範囲端


def test_format_dry_run_and_csv_handle_camera_none(tmp_path):
    r = report.build_report(
        target="bone", camera=None, bones={"センター": (5, 2)},
        selected_bones={"センター"}, ranges=[(0, 4)], keep_frames=[],
    )
    text = report.format_dry_run(r)  # 例外を出さない
    assert "センター" in text
    path = tmp_path / "p.csv"
    report.write_csv(r, str(path))  # camera 行なしでも書ける
    flat = path.read_text(encoding="utf-8")
    assert "センター" in flat


def test_write_json_roundtrips_and_keeps_japanese_unescaped(tmp_path):
    r = sample_report()
    path = tmp_path / "report.json"
    report.write_json(r, str(path))
    raw = path.read_text(encoding="utf-8")
    # 日本語は \uXXXX エスケープされず生で保持(ensure_ascii=False)。生テキストで確認する。
    assert "センター" in raw and "\\u30bb" not in raw
    loaded = json.loads(raw)
    assert loaded["target"] == "all"
    assert loaded["camera"]["output_keys"] == 2
    names = {b["name"] for b in loaded["bones"]}
    assert "センター" in names and "頭" in names


def test_write_csv_header_and_rows(tmp_path):
    r = sample_report()
    path = tmp_path / "preview.csv"
    report.write_csv(r, str(path))
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    header = rows[0]
    # ヘッダにトラック名・入出力キー数・削減率の列がある。
    assert "track" in header
    assert any("input" in h for h in header) and any("output" in h for h in header)
    ncols = len(header)
    # 全行が同じ列数(列ずれが無い)。
    assert all(len(row) == ncols for row in rows)
    tracks = {row[header.index("track")] for row in rows[1:]}
    assert "camera" in tracks and "センター" in tracks and "頭" in tracks


def test_write_json_bad_dir_raises(tmp_path):
    # 親ディレクトリ不在 → 例外(CLI は終了コード3にマップ)。
    with pytest.raises(OSError):
        report.write_json(sample_report(), str(tmp_path / "nodir" / "r.json"))


def test_write_csv_bad_dir_raises(tmp_path):
    with pytest.raises(OSError):
        report.write_csv(sample_report(), str(tmp_path / "nodir" / "p.csv"))
