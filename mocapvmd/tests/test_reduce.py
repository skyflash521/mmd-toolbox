"""mocapvmd インプロセス疎化のオーケストレーションのテスト(mocapvmd.md §3.3 / §5.3、実装計画 §3.5)。

reduce_bones はクリーニング後の全密ボーントラックを、種別ごとに解決した許容誤差で
mmd_toolbox.vmd.reduce.reduce_bone_track により疎化する(全範囲・全ボーン・カット検出あり)。本テストは
束ね方(種別別トレランス解決 + 固定引数の伝播 + curve_mode/override の受け渡し)を、建材を直接呼んだ
結果と突き合わせて検証する。疎化アルゴリズムそのものは mmd_toolbox 側のテストに委ねる。
"""

import pytest

from mmd_toolbox.vmd.reduce import build_bone_tolerances, measure_bone_errors, reduce_bone_track

from mocapvmd import presets
from mocapvmd import reduce as mreduce

from .helpers import BONE_NONLINEAR, bone

# reduce_bone_track へ渡す固定引数(設計確定値)。全範囲・カット検出あり。
_FIXED = dict(
    cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=False,
    min_seg=1, max_seg=180, strict=False,
)


def _curved(name, n=11, k=0.01):
    # ゆるく曲がる密トラック(疎化で中間キーが省かれる)。
    return [bone(name, f, pos=(round(k * f * f, 6), 0.0, 0.0)) for f in range(n)]


def _expected_tols(preset, category, **override):
    t = presets.resolve_reduction_tolerances(preset, category, **override)
    return build_bone_tolerances(t["bone_pos"], t["bone_rot"])


def _manual(track_keys, preset, category, curve_mode="bezier", override_pos=None, override_rot=None):
    t = presets.resolve_reduction_tolerances(
        preset, category, override_pos=override_pos, override_rot=override_rot
    )
    tols = build_bone_tolerances(t["bone_pos"], t["bone_rot"])
    f0, f1 = track_keys[0].frame, track_keys[-1].frame
    return reduce_bone_track(track_keys, [(f0, f1)], tols, curve_mode=curve_mode, **_FIXED)


def _track(out, name):
    return sorted((k for k in out if k.name == name), key=lambda k: k.frame)


def test_single_track_matches_building_blocks():
    keys = _curved("右足ＩＫ")
    out = mreduce.reduce_bones(keys, "balanced")
    expected = _manual(keys, "balanced", "foot_ik")
    assert _track(out, "右足ＩＫ") == list(expected)


def test_per_category_tolerance_resolution():
    # 種別ごとに解決した許容誤差(center 0.7 / fingers 1.5 スケール)で各トラックが疎化される。
    center = _curved("センター")
    fingers = _curved("右人指1")
    out = mreduce.reduce_bones(center + fingers, "balanced")
    assert _track(out, "センター") == list(_manual(center, "balanced", "center"))
    assert _track(out, "右人指1") == list(_manual(fingers, "balanced", "fingers"))
    # 余分なキー・重複トラックを出さない: 全出力が各トラックの疎化結果の結合と一致する。
    expected_all = list(_manual(center, "balanced", "center")) + list(_manual(fingers, "balanced", "fingers"))
    assert sorted(out, key=lambda k: (k.name, k.frame)) == sorted(expected_all, key=lambda k: (k.name, k.frame))


def test_per_track_range_tolerance_and_fixed_args(monkeypatch):
    # 各トラックに固有の範囲(実在区間)・種別別トレランス・固定引数が個別に渡ることを spy で固定する。
    # フレーム範囲の異なる2トラックを使い、範囲をまとめて全トラックへ誤流用する実装を弾く。
    calls = {}

    def _spy(source_keys, ranges, tols, **kw):
        calls[source_keys[0].name] = (ranges, tols, kw)
        return list(source_keys)

    monkeypatch.setattr(mreduce, "reduce_bone_track", _spy)
    center = _curved("センター")  # frames 0-10
    fingers = [bone("右人指1", f, pos=(round(0.01 * f * f, 6), 0.0, 0.0)) for f in range(5, 13)]  # 5-12
    mreduce.reduce_bones(center + fingers, "balanced")

    assert calls["センター"][0] == [(0, 10)]       # 各トラック固有の実在区間
    assert calls["右人指1"][0] == [(5, 12)]
    assert calls["センター"][1] == _expected_tols("balanced", "center")  # 種別別トレランス
    assert calls["右人指1"][1] == _expected_tols("balanced", "fingers")
    for name in ("センター", "右人指1"):
        kw = calls[name][2]
        for key, val in {
            "cut_thresholds": (1.0, 30.0), "keep_frames": [], "no_cut_detect": False,
            "min_seg": 1, "max_seg": 180, "strict": False, "curve_mode": "bezier",
        }.items():
            assert kw[key] == val


def test_override_and_curve_mode_flow_into_call(monkeypatch):
    # override は解決トレランスへ、curve_mode は reduce_bone_track へ伝播する(spy で固定)。
    captured = {}

    def _spy(source_keys, ranges, tols, **kw):
        captured["tols"] = tols
        captured["curve_mode"] = kw["curve_mode"]
        return list(source_keys)

    monkeypatch.setattr(mreduce, "reduce_bone_track", _spy)
    mreduce.reduce_bones(_curved("センター"), "balanced", override_pos=0.005, override_rot=0.05, curve_mode="linear")
    assert captured["tols"] == _expected_tols("balanced", "center", override_pos=0.005, override_rot=0.05)
    assert captured["curve_mode"] == "linear"


def test_single_key_track_kept_verbatim_alongside_multikey():
    # 多キートラックと同時入力でも、キー1個のトラックは逐語透過(非線形補間も保持)し、多キーは疎化される。
    # 入力全体が1キーのときだけ逐語返す誤実装を弾く。
    single = [bone("右足ＩＫ", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)]
    multi = _curved("センター")
    out = mreduce.reduce_bones(single + multi, "balanced")
    assert _track(out, "右足ＩＫ") == single
    assert _track(out, "センター") == list(_manual(multi, "balanced", "center"))


def test_deterministic():
    keys = _curved("右足ＩＫ")
    assert mreduce.reduce_bones(keys, "balanced") == mreduce.reduce_bones(keys, "balanced")


# --- 疎化診断(diagnostics_out。レポート §4.4 の削減率・適用許容・カット数・最大再生誤差の素データ) ---

_diag_pending = pytest.mark.xfail(reason="impl pending: Step 5f-1", strict=True)


def _expected_cuts(track_keys, tol):
    diag = {}
    reduce_bone_track(
        track_keys, [(track_keys[0].frame, track_keys[-1].frame)],
        build_bone_tolerances(tol["bone_pos"], tol["bone_rot"]), diagnostics=diag, **_FIXED,
    )
    return len(diag["cuts"])


@_diag_pending
def test_diagnostics_out_filled_per_track():
    # diagnostics_out に各トラックの入出力キー数・適用許容(種別別)・カット数・最大再生誤差を埋める。
    center = _curved("センター")     # frames 0-10
    fingers = _curved("右人指1")
    diag = {}
    out = mreduce.reduce_bones(center + fingers, "balanced", diagnostics_out=diag)
    assert set(diag) == {"センター", "右人指1"}
    for name, cat in (("センター", "center"), ("右人指1", "fingers")):
        src = _track(center + fingers, name)
        reduced = _track(out, name)
        tol = presets.resolve_reduction_tolerances("balanced", cat)
        d = diag[name]
        assert d["input_keys"] == len(src)
        assert d["output_keys"] == len(reduced)
        assert d["tol_pos"] == tol["bone_pos"]
        assert d["tol_rot"] == tol["bone_rot"]
        assert d["cuts"] == _expected_cuts(src, tol)
        assert d["errors"] == measure_bone_errors(src, reduced, [(src[0].frame, src[-1].frame)])


@_diag_pending
def test_diagnostics_out_override_tolerance():
    # override は適用許容(tol_pos/tol_rot)へ反映される(基準値上書き×種別スケール)。
    keys = _curved("センター")
    diag = {}
    mreduce.reduce_bones(keys, "balanced", override_pos=0.005, override_rot=0.05, diagnostics_out=diag)
    tol = presets.resolve_reduction_tolerances("balanced", "center", override_pos=0.005, override_rot=0.05)
    assert diag["センター"]["tol_pos"] == tol["bone_pos"]
    assert diag["センター"]["tol_rot"] == tol["bone_rot"]


@_diag_pending
def test_diagnostics_out_single_key_track_alongside_multikey():
    # 多キートラックと混在しても、キー1個(逐語透過)のトラックを削減なし(入出力1・カット0・誤差0・
    # 適用許容は種別の解決値)として診断に載せる(混在時に単一キーを診断から落とす実装を弾く)。
    single = [bone("右足ＩＫ", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)]
    multi = _curved("センター")
    diag = {}
    out = mreduce.reduce_bones(single + multi, "balanced", diagnostics_out=diag)
    assert set(diag) == {"右足ＩＫ", "センター"}
    assert _track(out, "右足ＩＫ") == single
    d = diag["右足ＩＫ"]
    tol = presets.resolve_reduction_tolerances("balanced", "foot_ik")
    assert d["input_keys"] == 1
    assert d["output_keys"] == 1
    assert d["tol_pos"] == tol["bone_pos"]
    assert d["tol_rot"] == tol["bone_rot"]
    assert d["cuts"] == 0
    assert d["errors"] == {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.0}


@_diag_pending
def test_diagnostics_out_does_not_change_output():
    # 診断収集は疎化結果を変えない。diagnostics_out 省略時(既定 None)と同一の出力を返す。
    keys = _curved("右足ＩＫ")
    diag = {}
    assert mreduce.reduce_bones(keys, "balanced", diagnostics_out=diag) == mreduce.reduce_bones(keys, "balanced")
