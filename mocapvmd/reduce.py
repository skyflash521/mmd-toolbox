"""mocapvmd インプロセス疎化のオーケストレーション(mocapvmd.md §3.3 / §5.3、実装計画 §3.5)。

クリーニング(一般ノイズ軽減)+足IK安定化の後、全密ボーントラックを mmd_toolbox.vmd.reduce の
共通機構で疎化する。種別ごとに resolve_reduction_tolerances で解決した許容誤差を build_bone_tolerances
で Tolerances 化し、reduce_bone_track へ渡す。全範囲・全ボーン・カット検出ありで、各トラックの疎化範囲は
トラック実在区間とする。キー1個以下のトラックは疎化できないため逐語透過する。
"""

from mmd_toolbox.vmd.reduce import build_bone_tolerances, measure_bone_errors, reduce_bone_track

from . import classify, presets

# reduce_bone_track へ渡す固定引数(全範囲・カット検出あり。sparsevmd の決め方を踏襲)。
_CUT_THRESHOLDS = (1.0, 30.0)  # (位置 MMD単位 / 回転 度)
_MIN_SEG = 1
_MAX_SEG = 180


_ZERO_ERRORS = {"pos_x": 0.0, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.0}


def reduce_bones(
    cleaned_keys, preset, *, override_pos=None, override_rot=None, curve_mode="bezier", diagnostics_out=None
):
    """クリーニング後の全密ボーントラックを種別別許容誤差で疎化し、疎なキー列を返す(§5.3)。

    名前ごとにトラック化し、種別別に解決した許容誤差(プリセット基準 × 種別スケール、override で基準
    上書き)で reduce_bone_track により疎化する。各トラックの範囲はトラック実在区間 [(first, last)]。
    キー1個以下のトラックは疎化できないため逐語保持する。

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

    out = []
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        category = classify.classify(name)
        if len(ks) <= 1:
            out.extend(ks)
            if diagnostics_out is not None:
                tol = presets.resolve_reduction_tolerances(
                    preset, category, override_pos=override_pos, override_rot=override_rot
                )
                diagnostics_out[name] = {
                    "input_keys": len(ks),
                    "output_keys": len(ks),
                    "tol_pos": tol["bone_pos"],
                    "tol_rot": tol["bone_rot"],
                    "cuts": 0,
                    "errors": dict(_ZERO_ERRORS),
                }
            continue
        tol = presets.resolve_reduction_tolerances(
            preset, category, override_pos=override_pos, override_rot=override_rot
        )
        tols = build_bone_tolerances(tol["bone_pos"], tol["bone_rot"])
        diag = {} if diagnostics_out is not None else None
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
        out.extend(reduced)
        if diagnostics_out is not None:
            f0, f1 = ks[0].frame, ks[-1].frame
            diagnostics_out[name] = {
                "input_keys": len(ks),
                "output_keys": len(reduced),
                "tol_pos": tol["bone_pos"],
                "tol_rot": tol["bone_rot"],
                "cuts": len(diag["cuts"]),
                "errors": measure_bone_errors(ks, reduced, [(f0, f1)]),
            }
    return out
