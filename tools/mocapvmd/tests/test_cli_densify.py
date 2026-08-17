"""mocapvmd の前段密化(疎キー+補間曲線入力)のテスト。

疎なキー(不等間隔可)+補間曲線の入力が、内部でMMD互換の補間評価により各トラックの
実在区間の密サンプルへ密化されてからクリーニング・疎化されることを検証する。
等価性(事前ベイクした密入力と同一結果)・--no-reduce の密キー出力・gap=1 の
値ジャンプ保持とカット保護・inspect 診断の密サンプル数を扱う。
"""

import json
import math

import pytest

from mocapvmd import cli
from vmd import interp, io
from vmd.reduce import BONE_LINEAR_INTERP, bone_interp_bytes

from .helpers import bone, write_vmd

# 非線形の補間曲線(x1,y1,x2,y2)。到達側キーに載せる。
_STEEP = (5, 122, 122, 5)
_EASE = (0, 64, 127, 64)
_NONLINEAR_ALL = bone_interp_bytes(_STEEP, _EASE, _STEEP, _EASE)


def _track_keys(doc, name):
    return sorted((k for k in doc.bone if k.name == name), key=lambda k: k.frame)


def _bake_input_tracks(doc):
    """入力ドキュメントの各トラック(キー2個以上)を実在区間で密ベイクしたキー列を返す。"""
    names = []
    seen = set()
    for k in doc.bone:
        if k.name not in seen:
            seen.add(k.name)
            names.append(k.name)
    out = []
    for name in names:
        ks = _track_keys(doc, name)
        if len(ks) < 2:
            out.extend(ks)
            continue
        positions, rotations = interp.bake_bone_track(ks, ks[0].frame, ks[-1].frame)
        for i, f in enumerate(range(ks[0].frame, ks[-1].frame + 1)):
            out.append(bone(name, f, pos=tuple(positions[i]), rot=tuple(rotations[i])))
    return out


# --- 等価性: 疎入力と事前ベイク済み密入力の処理結果が一致する ----------------


def test_full_pipeline_equivalence_sparse_vs_prebaked_dense(tmp_path):
    # 線形曲線+二進で正確な値なら事前ベイクは量子化誤差なしで書き戻せるため、
    # 全段(ノイズ軽減・足IK安定化・疎化)の結果が疎入力とバイト単位で一致する。
    # キー間隔は 2・4・8 の不等間隔。右足ＩＫは接地高さ(y=0)の遅い水平ドリフトで、
    # 足IK接地安定化の対象トラックとして密化後の信号に接地ロックが掛かる。
    sparse = tmp_path / "sparse.vmd"
    dense = tmp_path / "dense.vmd"
    out_a = tmp_path / "out_a.vmd"
    out_b = tmp_path / "out_b.vmd"
    frames_vals = [(0, 0.0), (2, 1.0), (6, 3.0), (14, 7.0)]
    write_vmd(sparse, bone=(
        [bone("センター", f, pos=(v, 0.5 * v, -v)) for f, v in frames_vals]
        + [bone("右腕", f, pos=(0.25 * v, 0.0, 0.0)) for f, v in frames_vals]
        + [bone("右足ＩＫ", f, pos=(0.125 * v, 0.0, 0.0)) for f, v in frames_vals]
    ))
    sparse_doc, _ = io.read(str(sparse))
    write_vmd(dense, bone=_bake_input_tracks(sparse_doc))

    assert cli.main([str(sparse), "-o", str(out_a)]) == 0
    assert cli.main([str(dense), "-o", str(out_b)]) == 0
    doc_a, _ = io.read(str(out_a))
    doc_b, _ = io.read(str(out_b))
    assert doc_a.bone == doc_b.bone


def test_cleaning_equivalence_nonlinear_curves(tmp_path):
    # 非線形曲線+不等間隔(gap 1・2・10・30)では事前ベイクの書き戻しに float32 量子化が
    # 乗るため、クリーニング(既定)+ --no-reduce の結果を許容差付きで比較する。
    sparse = tmp_path / "sparse.vmd"
    dense = tmp_path / "dense.vmd"
    out_a = tmp_path / "out_a.vmd"
    out_b = tmp_path / "out_b.vmd"
    frames_vals = [(0, 0.0), (1, 0.2), (3, 0.05), (13, 0.25), (43, 0.1)]
    write_vmd(sparse, bone=[
        bone("センター", f, pos=(v, -v, 2.0 * v),
             rot=(0.0, 0.0, math.sin(0.5 * v), math.cos(0.5 * v)), interp=_NONLINEAR_ALL)
        for f, v in frames_vals
    ])
    sparse_doc, _ = io.read(str(sparse))
    write_vmd(dense, bone=_bake_input_tracks(sparse_doc))

    assert cli.main([str(sparse), "-o", str(out_a), "--no-reduce"]) == 0
    assert cli.main([str(dense), "-o", str(out_b), "--no-reduce"]) == 0
    doc_a, _ = io.read(str(out_a))
    doc_b, _ = io.read(str(out_b))
    keys_a = _track_keys(doc_a, "センター")
    keys_b = _track_keys(doc_b, "センター")
    assert [k.frame for k in keys_a] == [k.frame for k in keys_b] == list(range(44))
    for ka, kb in zip(keys_a, keys_b, strict=True):
        assert ka.position == pytest.approx(kb.position, abs=1e-4)
        assert ka.rotation == pytest.approx(kb.rotation, abs=1e-4)


# --- --no-reduce: 疎入力からトラック実在区間ぶんの密キー(線形補間)が出る ----


def test_no_reduce_outputs_dense_linear_keys(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=(
        [bone("センター", 0, pos=(0.0, 0.0, 0.0)),
         bone("センター", 2, pos=(1.0, 0.0, 0.0), rot=(0.0, 0.0, 0.3, 0.9),
              interp=_NONLINEAR_ALL),
         bone("センター", 10, pos=(0.5, 1.0, 0.0), rot=(0.0, 0.3, 0.0, 0.9),
              interp=_NONLINEAR_ALL)]
        + [bone("右腕", 5, pos=(0.0, 0.0, 0.0)),
           bone("右腕", 8, pos=(0.4, 0.0, 0.0), rot=(0.3, 0.0, 0.0, 0.9),
                interp=_NONLINEAR_ALL)]
    ))
    code = cli.main([str(src), "-o", str(out),
                     "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"])
    assert code == 0
    in_doc, _ = io.read(str(src))
    out_doc, _ = io.read(str(out))
    # 各トラックの実在区間の全整数フレームに線形補間のキーが出る
    assert [k.frame for k in _track_keys(out_doc, "センター")] == list(range(11))
    assert [k.frame for k in _track_keys(out_doc, "右腕")] == list(range(5, 9))
    for k in out_doc.bone:
        assert k.interpolation == BONE_LINEAR_INTERP
    # 値は入力の補間曲線の評価値(float32 書き戻しの量子化差以内)
    for name in ("センター", "右腕"):
        in_keys = _track_keys(in_doc, name)
        positions, rotations = interp.bake_bone_track(in_keys, in_keys[0].frame, in_keys[-1].frame)
        for i, k in enumerate(_track_keys(out_doc, name)):
            assert k.position == pytest.approx(tuple(positions[i]), abs=1e-5)
            assert k.rotation == pytest.approx(tuple(rotations[i]), abs=1e-5)


# --- gap=1 の値ジャンプ: 密化後も保持され、疎化のカット保護に接続される ------


def test_gap1_value_jump_survives_reduce(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    # フレーム 5→6 が gap=1 でカット閾値(位置 1.0)超の値ジャンプ。前後は静止。
    write_vmd(src, bone=[
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 5, pos=(0.0, 0.0, 0.0)),
        bone("センター", 6, pos=(3.0, 0.0, 0.0)),
        bone("センター", 12, pos=(3.0, 0.0, 0.0)),
    ])
    code = cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize"])
    assert code == 0
    out_doc, _ = io.read(str(out))
    keys = {k.frame: k for k in _track_keys(out_doc, "センター")}
    # カット境界の両側キーが値そのままで残る(ジャンプが補間で丸められない)
    assert keys[5].position == pytest.approx((0.0, 0.0, 0.0))
    assert keys[6].position == pytest.approx((3.0, 0.0, 0.0))


# --- 重複キー: 同一フレームは後に現れたキーを採用する ------------------------


def test_duplicate_frame_keys_take_last_occurrence(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    # フレーム0が3件重複(先頭フレームの重複も含む)。採用されるのは各フレームの最後の1件。
    write_vmd(src, bone=[
        bone("センター", 0, pos=(9.0, 0.0, 0.0)),
        bone("センター", 0, pos=(5.0, 0.0, 0.0)),
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 4, pos=(1.0, 0.0, 0.0)),
        bone("センター", 4, pos=(2.0, 0.0, 0.0)),
    ])
    code = cli.main([str(src), "-o", str(out),
                     "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"])
    assert code == 0
    out_doc, _ = io.read(str(out))
    keys = _track_keys(out_doc, "センター")
    assert [k.frame for k in keys] == list(range(5))  # 重複が潰れて各フレーム1キー
    assert keys[0].position == pytest.approx((0.0, 0.0, 0.0))  # 後勝ち(9.0/5.0 でない)
    assert keys[4].position == pytest.approx((2.0, 0.0, 0.0))  # 後勝ち(1.0 でない)


# --- inspect 診断: reduction の input_keys は疎化に渡した密サンプル数 --------


def test_machine_inspect_reduction_input_keys_is_dense_count(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 10, pos=(1.0, 0.0, 0.0)),
    ])
    rc = cli.main([str(src), "--machine", "--dry-run",
                   "--no-denoise", "--no-foot-ik-stabilize"])
    assert rc == 0
    text = capsysbinary.readouterr().out.decode("utf-8")
    events = [json.loads(ln) for ln in text.split("\n") if ln]
    r = events[-1]
    assert r["type"] == "result" and r["mode"] == "inspect"
    # 入力由来フィールドは入力キー基準のまま
    assert r["keys"] == 2
    assert r["bones"][0]["keys"] == 2
    assert r["frame_range"] == [0, 10]
    # reduction の input_keys は密化後のサンプル数
    assert r["reduction"]["センター"]["input_keys"] == 11
