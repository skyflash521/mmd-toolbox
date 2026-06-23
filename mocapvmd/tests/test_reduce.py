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


def _expected_cuts(track_keys, tol):
    diag = {}
    reduce_bone_track(
        track_keys, [(track_keys[0].frame, track_keys[-1].frame)],
        build_bone_tolerances(tol["bone_pos"], tol["bone_rot"]), diagnostics=diag, **_FIXED,
    )
    return len(diag["cuts"])


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


def test_diagnostics_out_override_tolerance():
    # override は適用許容(tol_pos/tol_rot)へ反映される(基準値上書き×種別スケール)。
    keys = _curved("センター")
    diag = {}
    mreduce.reduce_bones(keys, "balanced", override_pos=0.005, override_rot=0.05, diagnostics_out=diag)
    tol = presets.resolve_reduction_tolerances("balanced", "center", override_pos=0.005, override_rot=0.05)
    assert diag["センター"]["tol_pos"] == tol["bone_pos"]
    assert diag["センター"]["tol_rot"] == tol["bone_rot"]


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


def test_diagnostics_out_does_not_change_output():
    # 診断収集は疎化結果を変えない。diagnostics_out 省略時(既定 None)と同一の出力を返す。
    keys = _curved("右足ＩＫ")
    diag = {}
    assert mreduce.reduce_bones(keys, "balanced", diagnostics_out=diag) == mreduce.reduce_bones(keys, "balanced")


# --- ボーン並列疎化(出力はシリアルと完全一致) ---
#
# 不変条件: ボーン単位の reduce は決定論的かつ実行順に非依存なので、並列化(ボーンを並行に
# reduce し固定順 first-seen へ再結合)しても reduce_bones の出力はシリアル(workers=1)と完全に
# 一致する。出力キー列・診断 diagnostics_out(値と first-seen 順)・ワーカ数によらない決定性を検証する。
#
# テストが触れる実装シーム:
#   reduce_bones(..., workers=N)   : ワーカ数(None=コア数基準, 1=シリアル)。
#   mreduce._MIN_PARALLEL_TRACKS   : 並列化する多キートラック数の下限(未満はシリアルへフォールバック)。
#   mreduce._make_pool(workers)    : プール生成シーム(発火・配分の検証で差し替える)。imap_unordered で配分。
# 並列実装が入るまでは xfail で印を付ける(strict で、実装後に通れば XPASS となり印の取り残しを検出する)。

_IMPL_PENDING = "impl pending: 並列reduce_bones"

# 種別ごとに許容値が異なるので、種別を混在させると per-bone 許容値を取り違える/共通化する誤実装を弾ける。
# 各接頭辞は分類で fingers / arms / legs / torso になり、fingers・torso・legs(=arms) で許容値が分かれる。
_PARALLEL_PREFIXES = ("右人指", "右腕", "右足", "上半身")


def _many_tracks(n_tracks, n=11):
    # 互いに曲率の異なる多キートラックを n_tracks 本。名前は一意で、種別(=許容値)が混在する。
    keys = []
    for i in range(n_tracks):
        name = f"{_PARALLEL_PREFIXES[i % len(_PARALLEL_PREFIXES)]}{i:02d}"
        keys.extend(_curved(name, n=n, k=0.01 + 0.001 * i))
    return keys


class _SpyPool:
    """_make_pool 差し替え用の疑似プール。配分数を記録し、ワーカ関数を完了順=入力の逆順で
    シリアル実行する。完了順に結果を連結して first-seen 順を壊す誤実装(順序復元の欠落)を弾く。"""

    def __init__(self, recorder):
        self._rec = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, items, chunksize=1):
        items = list(items)
        self._rec["dispatched"] = len(items)
        return [func(it) for it in reversed(items)]


@pytest.mark.xfail(reason=_IMPL_PENDING, strict=True)
def test_parallel_output_matches_serial_exact():
    # 実プロセス並列(workers=2)の出力キー列がシリアル(workers=1)と完全一致する。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)  # 閾値超で並列経路を確実に通す
    parallel = mreduce.reduce_bones(keys, "balanced", workers=2)
    serial = mreduce.reduce_bones(keys, "balanced", workers=1)
    assert parallel == serial


@pytest.mark.xfail(reason=_IMPL_PENDING, strict=True)
def test_parallel_diagnostics_match_serial_exact():
    # 診断データ(カット数・誤差・入出力キー数・適用許容)が並列とシリアルで、値も first-seen 順も一致する。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    d_par, d_ser = {}, {}
    mreduce.reduce_bones(keys, "balanced", workers=2, diagnostics_out=d_par)
    mreduce.reduce_bones(keys, "balanced", workers=1, diagnostics_out=d_ser)
    assert list(d_par.items()) == list(d_ser.items())


@pytest.mark.xfail(reason=_IMPL_PENDING, strict=True)
def test_workers_one_equals_default_serial():
    # workers=1 は workers 既定(省略)と同一出力(後方互換: 既存呼び出しを変えない)。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    assert mreduce.reduce_bones(keys, "balanced", workers=1) == mreduce.reduce_bones(keys, "balanced")


@pytest.mark.xfail(reason=_IMPL_PENDING, strict=True)
def test_parallel_fires_at_threshold_with_worker_count(monkeypatch):
    # 発火: 多キートラック数が閾値ちょうど・workers>1 でプールが指定ワーカ数で生成され全トラックが配分される。
    # spy は完了順を逆順で返すので、出力キー列・診断の両方が first-seen 順へ復元されることを決定的に検証する
    # (完了順のまま格納して first-seen 順を壊す誤実装を、出力・診断の双方で弾く)。
    rec = {}

    def fake_make_pool(workers):
        rec["workers"] = workers
        return _SpyPool(rec)

    monkeypatch.setattr(mreduce, "_make_pool", fake_make_pool)
    n = mreduce._MIN_PARALLEL_TRACKS  # 閾値ちょうど(>= で並列)
    keys = _many_tracks(n)
    d_par, d_ser = {}, {}
    out = mreduce.reduce_bones(keys, "balanced", workers=3, diagnostics_out=d_par)
    assert rec["workers"] == 3
    assert rec["dispatched"] == n  # 全多キートラックがワーカへ配分される
    serial = mreduce.reduce_bones(keys, "balanced", workers=1, diagnostics_out=d_ser)
    assert out == serial  # 逆順返却でも出力キーは first-seen 順に一致
    assert list(d_par.items()) == list(d_ser.items())  # 診断も first-seen 順へ復元される


@pytest.mark.xfail(reason=_IMPL_PENDING, strict=True)
def test_below_threshold_runs_serial(monkeypatch):
    # 閾値未満の小入力はプールを作らずシリアルにフォールバックする(並列オーバーヘッド回避)。
    made = {"pool": False}

    def fake_make_pool(workers):
        made["pool"] = True
        return _SpyPool({})

    monkeypatch.setattr(mreduce, "_make_pool", fake_make_pool)
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS - 1)  # 閾値未満
    out = mreduce.reduce_bones(keys, "balanced", workers=4)
    assert made["pool"] is False
    assert out == mreduce.reduce_bones(keys, "balanced", workers=1)


@pytest.mark.xfail(reason=_IMPL_PENDING, strict=True)
def test_threshold_counts_reducible_tracks_only(monkeypatch):
    # 閾値は疎化対象(多キー)トラック数で判定する。単一キートラックは逐語透過で配分対象外なので、
    # 総トラック数が閾値以上でも多キーが閾値未満ならシリアル(プール未生成)になる。閾値判定に
    # 「全トラック数」を誤用する実装を弾く。
    made = {"pool": False}

    def fake_make_pool(workers):
        made["pool"] = True
        return _SpyPool({})

    monkeypatch.setattr(mreduce, "_make_pool", fake_make_pool)
    multi = _many_tracks(mreduce._MIN_PARALLEL_TRACKS - 1)  # 多キーは閾値未満
    singles = [bone(f"単{i:02d}", 0, pos=(float(i), 0.0, 0.0)) for i in range(mreduce._MIN_PARALLEL_TRACKS)]
    keys = multi + singles  # 総トラック数は閾値以上だが、多キーは閾値未満
    out = mreduce.reduce_bones(keys, "balanced", workers=4)
    assert made["pool"] is False
    assert out == mreduce.reduce_bones(keys, "balanced", workers=1)


@pytest.mark.xfail(reason=_IMPL_PENDING, strict=True)
def test_parallel_interleaves_single_key_tracks_in_first_seen_order(monkeypatch):
    # 並列発火時(多キーが閾値以上)に、間に挟まった単一キー(逐語透過)トラックも first-seen 位置を保ち、
    # 出力列・診断 first-seen 順がシリアルと一致する。spy は多キーを逆順返却するので、単一キーをまとめて
    # 先頭/末尾へ別結合する誤実装や、多キーの順序復元漏れを決定的に弾く。
    rec = {}
    monkeypatch.setattr(mreduce, "_make_pool", lambda workers: _SpyPool(rec))
    keys = []
    for i in range(mreduce._MIN_PARALLEL_TRACKS):  # 多キーは閾値ちょうど(並列発火)
        name = f"{_PARALLEL_PREFIXES[i % len(_PARALLEL_PREFIXES)]}{i:02d}"
        keys.extend(_curved(name, k=0.01 + 0.001 * i))
        keys.append(bone(f"単{i:02d}", 0, pos=(float(i), 0.0, 0.0)))  # 各多キーの直後に単一キーを挟む
    d_par, d_ser = {}, {}
    out = mreduce.reduce_bones(keys, "balanced", workers=3, diagnostics_out=d_par)
    serial = mreduce.reduce_bones(keys, "balanced", workers=1, diagnostics_out=d_ser)
    assert out == serial
    assert list(d_par.items()) == list(d_ser.items())
