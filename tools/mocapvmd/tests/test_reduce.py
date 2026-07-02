"""mocapvmd インプロセス疎化のオーケストレーションのテスト(mocapvmd.md §3.3 / §5.3)。

reduce_bones はクリーニング後の全密ボーントラックを、種別ごとに解決した許容誤差で
vmd.reduce.reduce_bone_track により疎化する(全範囲・全ボーン・カット検出あり)。本テストは
束ね方(種別別トレランス解決 + 固定引数の伝播 + curve_mode/override の受け渡し)を、建材を直接呼んだ
結果と突き合わせて検証する。疎化アルゴリズムそのものは vmd 側のテストに委ねる。
"""

import pytest

from vmd.reduce import build_bone_tolerances, measure_bone_errors, reduce_bone_track

from mocapvmd import presets
from mocapvmd import reduce as mreduce

from .helpers import BONE_NONLINEAR, bone

# reduce_bone_track へ渡す固定引数(設計確定値)。全範囲・カット検出あり。
# max_seg は None=上限なし(無制限、機械的 cap 無効化)。固定引数だが None は他の引数と分けて各ヘルパで渡す。
_FIXED = dict(
    cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=False,
    min_seg=1, strict=False,
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
    return reduce_bone_track(track_keys, [(f0, f1)], tols, curve_mode=curve_mode, max_seg=None, **_FIXED)


def _track(out, name):
    return sorted((k for k in out if k.name == name), key=lambda k: k.frame)


def test_single_track_matches_building_blocks():
    keys = _curved("右足ＩＫ")
    out = mreduce.reduce_bones(keys, "medium")
    expected = _manual(keys, "medium", "foot_ik")
    assert _track(out, "右足ＩＫ") == list(expected)


def test_per_category_tolerance_resolution():
    # 種別ごとに解決した許容誤差(center 0.7 / fingers 1.5 スケール)で各トラックが疎化される。
    center = _curved("センター")
    fingers = _curved("右人指1")
    out = mreduce.reduce_bones(center + fingers, "medium")
    assert _track(out, "センター") == list(_manual(center, "medium", "center"))
    assert _track(out, "右人指1") == list(_manual(fingers, "medium", "fingers"))
    # 余分なキー・重複トラックを出さない: 全出力が各トラックの疎化結果の結合と一致する。
    expected_all = list(_manual(center, "medium", "center")) + list(_manual(fingers, "medium", "fingers"))
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
    mreduce.reduce_bones(center + fingers, "medium")

    assert calls["センター"][0] == [(0, 10)]       # 各トラック固有の実在区間
    assert calls["右人指1"][0] == [(5, 12)]
    assert calls["センター"][1] == _expected_tols("medium", "center")  # 種別別トレランス
    assert calls["右人指1"][1] == _expected_tols("medium", "fingers")
    for name in ("センター", "右人指1"):
        kw = calls[name][2]
        for key, val in {
            "cut_thresholds": (1.0, 30.0), "keep_frames": [], "no_cut_detect": False,
            "min_seg": 1, "strict": False, "curve_mode": "bezier",
        }.items():
            assert kw[key] == val
        # max_seg は None=上限なし(無制限)。機械的 cap を無効化し区間長で分割しない。
        assert kw["max_seg"] is None


def test_override_and_curve_mode_flow_into_call(monkeypatch):
    # override は解決トレランスへ、curve_mode は reduce_bone_track へ伝播する(spy で固定)。
    captured = {}

    def _spy(source_keys, ranges, tols, **kw):
        captured["tols"] = tols
        captured["curve_mode"] = kw["curve_mode"]
        return list(source_keys)

    monkeypatch.setattr(mreduce, "reduce_bone_track", _spy)
    mreduce.reduce_bones(_curved("センター"), "medium", override_pos=0.005, override_rot=0.05, curve_mode="linear")
    assert captured["tols"] == _expected_tols("medium", "center", override_pos=0.005, override_rot=0.05)
    assert captured["curve_mode"] == "linear"


def test_single_key_track_kept_verbatim_alongside_multikey():
    # 多キートラックと同時入力でも、キー1個のトラックは逐語透過(非線形補間も保持)し、多キーは疎化される。
    # 入力全体が1キーのときだけ逐語返す誤実装を弾く。
    single = [bone("右足ＩＫ", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)]
    multi = _curved("センター")
    out = mreduce.reduce_bones(single + multi, "medium")
    assert _track(out, "右足ＩＫ") == single
    assert _track(out, "センター") == list(_manual(multi, "medium", "center"))


def test_deterministic():
    keys = _curved("右足ＩＫ")
    assert mreduce.reduce_bones(keys, "medium") == mreduce.reduce_bones(keys, "medium")


# --- 疎化診断(diagnostics_out。レポート §4.4 の削減率・適用許容・カット数・最大再生誤差の素データ) ---


def _expected_cuts(track_keys, tol):
    diag = {}
    f0, f1 = track_keys[0].frame, track_keys[-1].frame
    reduce_bone_track(
        track_keys, [(f0, f1)],
        build_bone_tolerances(tol["bone_pos"], tol["bone_rot"]),
        diagnostics=diag, max_seg=None, **_FIXED,
    )
    return len(diag["cuts"])


def test_diagnostics_out_filled_per_track():
    # diagnostics_out に各トラックの入出力キー数・適用許容(種別別)・カット数・最大再生誤差を埋める。
    center = _curved("センター")     # frames 0-10
    fingers = _curved("右人指1")
    diag = {}
    out = mreduce.reduce_bones(center + fingers, "medium", diagnostics_out=diag)
    assert set(diag) == {"センター", "右人指1"}
    for name, cat in (("センター", "center"), ("右人指1", "fingers")):
        src = _track(center + fingers, name)
        reduced = _track(out, name)
        tol = presets.resolve_reduction_tolerances("medium", cat)
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
    mreduce.reduce_bones(keys, "medium", override_pos=0.005, override_rot=0.05, diagnostics_out=diag)
    tol = presets.resolve_reduction_tolerances("medium", "center", override_pos=0.005, override_rot=0.05)
    assert diag["センター"]["tol_pos"] == tol["bone_pos"]
    assert diag["センター"]["tol_rot"] == tol["bone_rot"]


def test_diagnostics_out_single_key_track_alongside_multikey():
    # 多キートラックと混在しても、キー1個(逐語透過)のトラックを削減なし(入出力1・カット0・誤差0・
    # 適用許容は種別の解決値)として診断に載せる(混在時に単一キーを診断から落とす実装を弾く)。
    single = [bone("右足ＩＫ", 0, pos=(1.0, 2.0, 3.0), interp=BONE_NONLINEAR)]
    multi = _curved("センター")
    diag = {}
    out = mreduce.reduce_bones(single + multi, "medium", diagnostics_out=diag)
    assert set(diag) == {"右足ＩＫ", "センター"}
    assert _track(out, "右足ＩＫ") == single
    d = diag["右足ＩＫ"]
    tol = presets.resolve_reduction_tolerances("medium", "foot_ik")
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
    assert mreduce.reduce_bones(keys, "medium", diagnostics_out=diag) == mreduce.reduce_bones(keys, "medium")


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


def test_parallel_output_matches_serial_exact():
    # 実プロセス並列(workers=2)の出力キー列がシリアル(workers=1)と完全一致する。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)  # 閾値超で並列経路を確実に通す
    parallel = mreduce.reduce_bones(keys, "medium", workers=2)
    serial = mreduce.reduce_bones(keys, "medium", workers=1)
    assert parallel == serial


def test_parallel_diagnostics_match_serial_exact():
    # 診断データ(カット数・誤差・入出力キー数・適用許容)が並列とシリアルで、値も first-seen 順も一致する。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    d_par, d_ser = {}, {}
    mreduce.reduce_bones(keys, "medium", workers=2, diagnostics_out=d_par)
    mreduce.reduce_bones(keys, "medium", workers=1, diagnostics_out=d_ser)
    assert list(d_par.items()) == list(d_ser.items())


def test_workers_one_equals_default_serial():
    # workers=1 は workers 既定(省略)と同一出力(後方互換: 既存呼び出しを変えない)。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    assert mreduce.reduce_bones(keys, "medium", workers=1) == mreduce.reduce_bones(keys, "medium")


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
    out = mreduce.reduce_bones(keys, "medium", workers=3, diagnostics_out=d_par)
    assert rec["workers"] == 3
    assert rec["dispatched"] == n  # 全多キートラックがワーカへ配分される
    serial = mreduce.reduce_bones(keys, "medium", workers=1, diagnostics_out=d_ser)
    assert out == serial  # 逆順返却でも出力キーは first-seen 順に一致
    assert list(d_par.items()) == list(d_ser.items())  # 診断も first-seen 順へ復元される


def test_below_threshold_runs_serial(monkeypatch):
    # 閾値未満の小入力はプールを作らずシリアルにフォールバックする(並列オーバーヘッド回避)。
    made = {"pool": False}

    def fake_make_pool(workers):
        made["pool"] = True
        return _SpyPool({})

    monkeypatch.setattr(mreduce, "_make_pool", fake_make_pool)
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS - 1)  # 閾値未満
    out = mreduce.reduce_bones(keys, "medium", workers=4)
    assert made["pool"] is False
    assert out == mreduce.reduce_bones(keys, "medium", workers=1)


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
    out = mreduce.reduce_bones(keys, "medium", workers=4)
    assert made["pool"] is False
    assert out == mreduce.reduce_bones(keys, "medium", workers=1)


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
    out = mreduce.reduce_bones(keys, "medium", workers=3, diagnostics_out=d_par)
    serial = mreduce.reduce_bones(keys, "medium", workers=1, diagnostics_out=d_ser)
    assert out == serial
    assert list(d_par.items()) == list(d_ser.items())


# --- 進捗コールバック(progress)。疎化の進行をボーン完了単位で親へ通知する副作用専用フック。 ---
#
# 計画 §4 / §6.1 の契約: reduce_bones に任意の progress=None を足し、疎化対象(多キー)トラックの確定時に
# progress(0, len(multikey))、各ボーン完了ごとに progress(done, total) を呼ぶ。total は常に多キー本数で、
# done は 0→total を1ずつ進む(単一キー逐語トラックは数えない)。並列(完了順 unordered)でも done は
# 単なる完了カウンタなので、(done, total) 列はシリアルと一致する。副作用専用で疎化結果は変えない。


class _GenSpyPool:
    """imap_unordered を遅延ジェネレータで返し、各 yield 時点を events に記録する疑似プール。
    親が完了(yield)ごとに progress を呼ぶか(=完了イベント単位通知)を yield と progress の
    interleave で検証するために使う。完了順は入力の逆順。"""

    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, func, items, chunksize=1):
        for i, it in enumerate(reversed(list(items))):
            self._events.append(("yield", i))
            yield func(it)


def test_progress_param_does_not_change_output():
    # progress を渡しても疎化結果(キー列・診断)は progress 省略時と完全一致する(副作用専用)。並列・シリアル両経路。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    for workers in (1, 2):
        d_cb, d_no = {}, {}
        with_cb = mreduce.reduce_bones(
            keys, "medium", workers=workers, progress=lambda d, t: None, diagnostics_out=d_cb
        )
        without = mreduce.reduce_bones(keys, "medium", workers=workers, diagnostics_out=d_no)
        assert with_cb == without
        assert list(d_cb.items()) == list(d_no.items())  # 診断も値・first-seen 順とも不変


def test_progress_contract_serial():
    # シリアル経路: total=多キー本数で固定、done は 0→total を1ずつ。混在する単一キーは数えない。
    # workers=1 と「workers>1 だが多キーが閾値未満でシリアルフォールバック」の両方で同じ列になることを固定
    # する(フォールバック経路で progress 通知が漏れる実装を弾く)。
    multi = mreduce._MIN_PARALLEL_TRACKS - 1  # 閾値未満でシリアル経路(workers>1 でもフォールバック)
    keys = _many_tracks(multi)
    keys += [bone(f"単{i:02d}", 0, pos=(float(i), 0.0, 0.0)) for i in range(3)]  # 単一キーは progress 対象外
    for workers in (1, 4):
        calls = []
        mreduce.reduce_bones(keys, "medium", workers=workers, progress=lambda d, t: calls.append((d, t)))
        assert calls == [(d, multi) for d in range(multi + 1)]  # (0,M),(1,M),…,(M,M)


def test_progress_reports_zero_total_when_no_multikey():
    # 疎化対象(多キー)が皆無でも、確定時通知 progress(0, 0) を1回だけ呼ぶ(total を確定させる)。
    # 「多キーがあるときだけ初回通知する」実装だと total 未確定のままになるのを弾く。
    keys = [bone(f"単{i:02d}", 0, pos=(float(i), 0.0, 0.0)) for i in range(5)]  # 全て単一キー
    calls = []
    mreduce.reduce_bones(keys, "medium", workers=4, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(0, 0)]


def test_progress_contract_parallel(monkeypatch):
    # 並列経路: (done, total) 列はシリアルと同一(total 固定・done は 0→total、単一キーは数えない)。加えて
    # 「imap_unordered の完了(yield)ごとに progress を呼ぶ」完了イベント単位通知を、yield と progress の
    # interleave で固定する。全ワーカ結果を溜め終えてから一括で progress を呼ぶ誤実装(末尾まで進捗が
    # 動かず停滞して見える)を弾く。
    events = []
    monkeypatch.setattr(mreduce, "_make_pool", lambda workers: _GenSpyPool(events))
    multi = mreduce._MIN_PARALLEL_TRACKS  # 閾値ちょうどで並列発火
    keys = _many_tracks(multi)
    keys += [bone(f"単{i:02d}", 0, pos=(float(i), 0.0, 0.0)) for i in range(3)]
    done_calls = []

    def _prog(d, t):
        events.append(("progress", d, t))
        done_calls.append((d, t))

    mreduce.reduce_bones(keys, "medium", workers=3, progress=_prog)
    assert done_calls == [(d, multi) for d in range(multi + 1)]  # total 固定・done 0→total
    # 確定時に progress(0,M)、以後 yield ごとに progress(done,M)。yield と progress が交互に並ぶ。
    expected = [("progress", 0, multi)]
    for i in range(multi):
        expected += [("yield", i), ("progress", i + 1, multi)]
    assert events == expected


# --- 並列ワーカの SIGINT 無視 initializer(mocapvmd.md §10.6) ---
#
# Windows の Ctrl-C(CTRL_C_EVENT)は同一コンソールの全プロセスへ配送されるため、ワーカが SIGINT で
# 任意位置で死ぬとトレースバックが漏れ、失われたタスクを親が待ち続けうる。ワーカに SIGINT を無視させ、
# 中断の畳み込みを親プロセスへ一元化する。中断→cancelled/130 の CLI 側の畳み込みは test_cli_machine が固定する。


def test_reduce_worker_init_ignores_sigint():
    # ワーカ initializer は SIGINT を SIG_IGN に設定する。親プロセスで直接呼び、getsignal で検証して復元する。
    import signal

    prev = signal.getsignal(signal.SIGINT)
    try:
        mreduce._reduce_worker_init()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, prev)


def test_make_pool_wires_sigint_initializer(monkeypatch):
    # _make_pool は initializer=_reduce_worker_init でプールを生成する(ワーカへ SIGINT 無視を配線)。
    captured = {}

    class _Ctx:
        def Pool(self, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(mreduce, "get_context", lambda method: _Ctx())
    mreduce._make_pool(3)
    assert captured["processes"] == 3
    assert captured["initializer"] is mreduce._reduce_worker_init


def test_parallel_progress_values_match_serial():
    # progress の (done, total) 値列が並列(workers=2)と逐次(workers=1)で一致する(完了順=並列では
    # 非決定だが done は 0→total で値列は不変。§10.6 の決定論)。
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)
    par, ser = [], []
    mreduce.reduce_bones(keys, "medium", workers=2, progress=lambda d, t: par.append((d, t)))
    mreduce.reduce_bones(keys, "medium", workers=1, progress=lambda d, t: ser.append((d, t)))
    assert par == ser
    total = mreduce._MIN_PARALLEL_TRACKS + 1
    assert ser[0] == (0, total) and ser[-1] == (total, total)  # (0,total) → (total,total)


def test_parallel_keyboard_interrupt_terminates_pool(monkeypatch):
    # 並列疎化ループ中の KeyboardInterrupt でプールが畳まれ(with ブロックの __exit__ が呼ばれ、実プールは
    # terminate)、中断が親へ伝播する(main が cancelled/130 へ畳む前提。§10.6)。with を外して pool を
    # 畳まない誤実装を弾く。
    exits = []

    class _InterruptingPool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, *rest):
            exits.append(exc_type)
            return False  # 例外は握り潰さず伝播させる

        def imap_unordered(self, func, items, chunksize=1):
            # 呼び出し時でなく反復中に中断する: with 内で iterator を作り反復を with 外へ出す誤実装も
            # 弾けるよう、yield を持つジェネレータにして最初の next() で投げる(反復=for ループ中の中断)。
            if False:
                yield None
            raise KeyboardInterrupt()

    monkeypatch.setattr(mreduce, "_make_pool", lambda workers: _InterruptingPool())
    keys = _many_tracks(mreduce._MIN_PARALLEL_TRACKS + 1)  # 閾値超で並列経路を確実に通す
    with pytest.raises(KeyboardInterrupt):
        mreduce.reduce_bones(keys, "medium", workers=2)
    assert exits == [KeyboardInterrupt]  # with __exit__ が中断で呼ばれた(= 実プールなら terminate)
