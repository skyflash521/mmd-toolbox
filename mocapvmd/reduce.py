"""mocapvmd インプロセス疎化のオーケストレーション(mocapvmd.md §3.3 / §5.3、実装計画 §3.5)。

クリーニング(一般ノイズ軽減)+足IK安定化の後、全密ボーントラックを mmd_toolbox.vmd.reduce の
共通機構で疎化する。種別ごとに resolve_reduction_tolerances で解決した許容誤差を build_bone_tolerances
で Tolerances 化し、reduce_bone_track へ渡す。全範囲・全ボーン・カット検出ありで、各トラックの疎化範囲は
トラック実在区間とする。キー1個以下のトラックは疎化できないため逐語透過する。
"""

from mmd_toolbox.vmd.reduce import build_bone_tolerances, reduce_bone_track

from . import classify, presets

# reduce_bone_track へ渡す固定引数(全範囲・カット検出あり。sparsevmd の決め方を踏襲)。
_CUT_THRESHOLDS = (1.0, 30.0)  # (位置 MMD単位 / 回転 度)
_MIN_SEG = 1
_MAX_SEG = 180


def reduce_bones(cleaned_keys, preset, *, override_pos=None, override_rot=None, curve_mode="bezier"):
    """クリーニング後の全密ボーントラックを種別別許容誤差で疎化し、疎なキー列を返す(§5.3)。

    名前ごとにトラック化し、種別別に解決した許容誤差(プリセット基準 × 種別スケール、override で基準
    上書き)で reduce_bone_track により疎化する。各トラックの範囲はトラック実在区間 [(first, last)]。
    キー1個以下のトラックは疎化できないため逐語保持する。
    """
    order = []
    groups = {}
    for k in cleaned_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    out = []
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        if len(ks) <= 1:
            out.extend(ks)
            continue
        category = classify.classify(name)
        tol = presets.resolve_reduction_tolerances(
            preset, category, override_pos=override_pos, override_rot=override_rot
        )
        tols = build_bone_tolerances(tol["bone_pos"], tol["bone_rot"])
        out.extend(
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
            )
        )
    return out
