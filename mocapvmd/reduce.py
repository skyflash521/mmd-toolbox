"""mocapvmd インプロセス疎化のオーケストレーション(mocapvmd.md §3.3 / §5.3、実装計画 §3.5)。

クリーニング(一般ノイズ軽減)+足IK安定化の後、全密ボーントラックを mmd_toolbox.vmd.reduce の
共通機構で疎化する。種別ごとに resolve_reduction_tolerances で解決した許容誤差を build_bone_tolerances
で Tolerances 化し、reduce_bone_track へ渡す。全範囲・全ボーン・カット検出ありで、各トラックの疎化範囲は
トラック実在区間とする。キー1個以下のトラックは疎化できないため逐語透過する。

ボーンは互いに独立に疎化でき(reduce_bone_track はボーンごとに閉じる)、per-bone reduce は決定論的で
実行順に依存しない。多キートラックが多い密入力では、これをプロセス並列で分散して実時間を短縮する。
再結合をトラックの first-seen 順で行うため、出力(キー列・診断)はワーカ数・完了順に依らずシリアルと
完全に一致する。
"""

import os
from multiprocessing import get_context

from mmd_toolbox.vmd.reduce import build_bone_tolerances, measure_bone_errors, reduce_bone_track

from . import classify, presets

# reduce_bone_track へ渡す固定引数(全範囲・カット検出あり。sparsevmd の決め方を踏襲)。
_CUT_THRESHOLDS = (1.0, 30.0)  # (位置 MMD単位 / 回転 度)
_MIN_SEG = 1
_MAX_SEG = 180

# 多キートラックがこの本数以上のときだけプロセス並列化する。未満はプール起動・データ転送のコストが
# 並列の利得を上回るのでシリアルにフォールバックする。判定は疎化対象(多キー)トラック数で行う
# (単一キーは逐語透過で配分しないため数えない)。
_MIN_PARALLEL_TRACKS = 8


_ZERO_ERRORS = {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.0}


def _reduce_track(ks, tol_pos, tol_rot, curve_mode, want_diag):
    """多キートラック ks を疎化し (疎キー列, 診断payload) を返す。reduce_bones のシリアル/並列の
    両経路が呼ぶ単一の per-bone 実体で、両経路の reduce を同一に保つ。診断payload は want_diag のとき
    {"cuts": カット数, "errors": 軸別最大再生誤差}、不要なら None。
    """
    tols = build_bone_tolerances(tol_pos, tol_rot)
    diag = {} if want_diag else None
    reduced = list(
        reduce_bone_track(
            ks,
            [(ks[0].frame, ks[-1].frame)],
            tols,
            cut_thresholds=_CUT_THRESHOLDS,
            keep_frames=[],
            no_cut_detect=False,
            min_seg=_MIN_SEG,
            max_seg=_MAX_SEG,
            strict=False,
            curve_mode=curve_mode,
            diagnostics=diag,
        )
    )
    if not want_diag:
        return reduced, None
    return reduced, {
        "cuts": len(diag["cuts"]),
        "errors": measure_bone_errors(ks, reduced, [(ks[0].frame, ks[-1].frame)]),
    }


def _reduce_one(item):
    """プロセスワーカ。pickle 可能な item から1ボーンを疎化し (name, 疎キー列, 診断payload) を返す。
    spawn(Windows)で子プロセスが再 import するため module-level に置く(closure/lambda は不可)。
    """
    name, ks, tol_pos, tol_rot, curve_mode, want_diag = item
    reduced, payload = _reduce_track(ks, tol_pos, tol_rot, curve_mode, want_diag)
    return name, reduced, payload


def _make_pool(workers):
    """ワーカ数 workers のプロセスプールを生成する。OS 既定に依らず spawn を明示し(Windows と同条件で
    pickle 可能性を担保)、プール生成を1か所に閉じ込める。
    """
    return get_context("spawn").Pool(processes=workers)


def _resolve_workers(workers):
    """ワーカ数を解決する。None は CPU コア数基準(取得不能時は1)、指定値は最低1にクランプする。"""
    if workers is None:
        return os.cpu_count() or 1
    return max(1, int(workers))


def reduce_bones(
    cleaned_keys, preset, *, override_pos=None, override_rot=None, curve_mode="bezier",
    diagnostics_out=None, workers=None,
):
    """クリーニング後の全密ボーントラックを種別別許容誤差で疎化し、疎なキー列を返す(§5.3)。

    名前ごとにトラック化し、種別別に解決した許容誤差(プリセット基準 × 種別スケール、override で基準
    上書き)で reduce_bone_track により疎化する。各トラックの範囲はトラック実在区間 [(first, last)]。
    キー1個以下のトラックは疎化できないため逐語保持する。

    workers で疎化のプロセス並列数を指定する(None=CPU コア数基準, 1=シリアル)。多キートラックが
    _MIN_PARALLEL_TRACKS 以上で workers>1 のときだけ並列化し、未満はシリアルへフォールバックする。
    各ボーンの reduce は決定論的で実行順に非依存、再結合をトラックの first-seen 順で行うため、出力は
    ワーカ数・完了順に依らずシリアル(workers=1)と完全に一致する。

    diagnostics_out に dict を渡すと、レポート(§4.4)用にトラックごとの素データ
    {input_keys, output_keys, tol_pos, tol_rot, cuts, errors} を埋める。cuts は reduce_bone_track の
    検出カット数、errors は measure_bone_errors の軸別最大再生誤差(疎化前の密 vs 疎化後)。キー1個以下の
    逐語トラックは削減なし(入出力同数・カット0・誤差0)として載せる。収集は疎化結果を変えない。
    """
    order = []
    groups = {}
    for k in cleaned_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    want_diag = diagnostics_out is not None
    sorted_tracks = {name: sorted(groups[name], key=lambda k: k.frame) for name in order}
    multikey = [name for name in order if len(sorted_tracks[name]) > 1]

    def _tol(name):
        return presets.resolve_reduction_tolerances(
            preset, classify.classify(name), override_pos=override_pos, override_rot=override_rot
        )

    # 多キートラックが閾値以上・workers>1 ならプロセス並列で先に疎化する。結果は name でひいて後段の
    # first-seen ループへ渡すので、ワーカの完了順に依存しない(出力・診断はシリアルと完全一致)。
    precomputed = {}
    n_workers = _resolve_workers(workers)
    if n_workers > 1 and len(multikey) >= _MIN_PARALLEL_TRACKS:
        work = []
        for name in multikey:
            tol = _tol(name)
            work.append((name, sorted_tracks[name], tol["bone_pos"], tol["bone_rot"], curve_mode, want_diag))
        with _make_pool(n_workers) as pool:
            for name, reduced, payload in pool.imap_unordered(_reduce_one, work, chunksize=1):
                precomputed[name] = (reduced, payload)

    out = []
    for name in order:
        ks = sorted_tracks[name]
        if len(ks) <= 1:
            out.extend(ks)
            if want_diag:
                tol = _tol(name)
                diagnostics_out[name] = {
                    "input_keys": len(ks),
                    "output_keys": len(ks),
                    "tol_pos": tol["bone_pos"],
                    "tol_rot": tol["bone_rot"],
                    "cuts": 0,
                    "errors": dict(_ZERO_ERRORS),
                }
            continue
        tol = _tol(name)
        if name in precomputed:
            reduced, payload = precomputed[name]
        else:
            reduced, payload = _reduce_track(ks, tol["bone_pos"], tol["bone_rot"], curve_mode, want_diag)
        out.extend(reduced)
        if want_diag:
            f0, f1 = ks[0].frame, ks[-1].frame
            diagnostics_out[name] = {
                "input_keys": len(ks),
                "output_keys": len(reduced),
                "tol_pos": tol["bone_pos"],
                "tol_rot": tol["bone_rot"],
                "cuts": payload["cuts"],
                "errors": payload["errors"],
            }
    return out
