import json
import math

import pytest

from mocapvmd import cli
from vmd import interp, io
from vmd.reduce import BONE_LINEAR_INTERP, bone_interp_bytes

from .helpers import bone, write_vmd

_STEEP = (5, 122, 122, 5)
_EASE = (0, 64, 127, 64)
_NONLINEAR_ALL = bone_interp_bytes(_STEEP, _EASE, _STEEP, _EASE)


def _track_keys(doc, name):
    return sorted((k for k in doc.bone if k.name == name), key=lambda k: k.frame)


def _bake_each_multi_key_track_over_its_key_span(doc):
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


def test_linear_dyadic_sparse_input_matches_prebaked_dense_exactly_through_full_pipeline(tmp_path):
    sparse = tmp_path / "sparse.vmd"
    dense = tmp_path / "dense.vmd"
    out_a = tmp_path / "out_a.vmd"
    out_b = tmp_path / "out_b.vmd"
    unequal_gap_dyadic_keys = [(0, 0.0), (2, 1.0), (6, 3.0), (14, 7.0)]
    write_vmd(sparse, bone=(
        [bone("センター", f, pos=(v, 0.5 * v, -v)) for f, v in unequal_gap_dyadic_keys]
        + [bone("右腕", f, pos=(0.25 * v, 0.0, 0.0)) for f, v in unequal_gap_dyadic_keys]
        + [bone("右足ＩＫ", f, pos=(0.125 * v, 0.0, 0.0)) for f, v in unequal_gap_dyadic_keys]
    ))
    sparse_doc, _ = io.read(str(sparse))
    write_vmd(dense, bone=_bake_each_multi_key_track_over_its_key_span(sparse_doc))

    assert cli.main([str(sparse), "-o", str(out_a)]) == 0
    assert cli.main([str(dense), "-o", str(out_b)]) == 0
    doc_a, _ = io.read(str(out_a))
    doc_b, _ = io.read(str(out_b))
    assert doc_a.bone == doc_b.bone


def test_nonlinear_sparse_input_matches_prebaked_dense_within_float32_after_cleaning(tmp_path):
    float32_writeback_tol = 1e-4
    sparse = tmp_path / "sparse.vmd"
    dense = tmp_path / "dense.vmd"
    out_a = tmp_path / "out_a.vmd"
    out_b = tmp_path / "out_b.vmd"
    unequal_gap_keys = [(0, 0.0), (1, 0.2), (3, 0.05), (13, 0.25), (43, 0.1)]
    write_vmd(sparse, bone=[
        bone("センター", f, pos=(v, -v, 2.0 * v),
             rot=(0.0, 0.0, math.sin(0.5 * v), math.cos(0.5 * v)), interp=_NONLINEAR_ALL)
        for f, v in unequal_gap_keys
    ])
    sparse_doc, _ = io.read(str(sparse))
    write_vmd(dense, bone=_bake_each_multi_key_track_over_its_key_span(sparse_doc))

    assert cli.main([str(sparse), "-o", str(out_a), "--no-reduce"]) == 0
    assert cli.main([str(dense), "-o", str(out_b), "--no-reduce"]) == 0
    doc_a, _ = io.read(str(out_a))
    doc_b, _ = io.read(str(out_b))
    keys_a = _track_keys(doc_a, "センター")
    keys_b = _track_keys(doc_b, "センター")
    assert [k.frame for k in keys_a] == [k.frame for k in keys_b] == list(range(44))
    for ka, kb in zip(keys_a, keys_b, strict=True):
        assert ka.position == pytest.approx(kb.position, abs=float32_writeback_tol)
        assert ka.rotation == pytest.approx(kb.rotation, abs=float32_writeback_tol)


def test_no_reduce_outputs_linear_key_per_frame_of_track_span_with_curve_values(tmp_path):
    float32_writeback_tol = 1e-5
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
    assert [k.frame for k in _track_keys(out_doc, "センター")] == list(range(11))
    assert [k.frame for k in _track_keys(out_doc, "右腕")] == list(range(5, 9))
    for k in out_doc.bone:
        assert k.interpolation == BONE_LINEAR_INTERP
    for name in ("センター", "右腕"):
        in_keys = _track_keys(in_doc, name)
        positions, rotations = interp.bake_bone_track(in_keys, in_keys[0].frame, in_keys[-1].frame)
        for i, k in enumerate(_track_keys(out_doc, name)):
            assert k.position == pytest.approx(tuple(positions[i]), abs=float32_writeback_tol)
            assert k.rotation == pytest.approx(tuple(rotations[i]), abs=float32_writeback_tol)


def test_gap1_jump_above_cut_threshold_keeps_both_boundary_values_after_reduce(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
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
    assert keys[5].position == pytest.approx((0.0, 0.0, 0.0))
    assert keys[6].position == pytest.approx((3.0, 0.0, 0.0))


def test_duplicate_frame_keys_collapse_to_one_per_frame_taking_last_occurrence(tmp_path):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
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
    assert [k.frame for k in keys] == list(range(5))
    assert keys[0].position == pytest.approx((0.0, 0.0, 0.0))
    assert keys[4].position == pytest.approx((2.0, 0.0, 0.0))


def _last_machine_event(capsysbinary):
    text = capsysbinary.readouterr().out.decode("utf-8")
    return [json.loads(ln) for ln in text.split("\n") if ln][-1]


def test_process_result_input_keys_counts_duplicate_frame_keys_before_collapse(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    input_keys = [
        bone("センター", 0, pos=(9.0, 0.0, 0.0)),
        bone("センター", 0, pos=(5.0, 0.0, 0.0)),
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 4, pos=(2.0, 0.0, 0.0)),
    ]
    write_vmd(src, bone=input_keys)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine",
                   "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"])
    assert rc == 0
    r = _last_machine_event(capsysbinary)
    assert r["mode"] == "process"
    assert r["input_keys"] == len(input_keys)


def test_non_finite_value_in_discarded_duplicate_key_is_invalid_bone_values(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[
        bone("センター", 0, pos=(float("nan"), 0.0, 0.0)),
        bone("センター", 0, pos=(0.0, 0.0, 0.0)),
        bone("センター", 4, pos=(2.0, 0.0, 0.0)),
    ])
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 1
    e = _last_machine_event(capsysbinary)
    assert e["type"] == "error" and e["code"] == "invalid_bone_values"


def test_machine_inspect_input_fields_count_input_keys_but_reduction_counts_dense_samples(
        tmp_path, capsysbinary):
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
    assert r["keys"] == 2
    assert r["bones"][0]["keys"] == 2
    assert r["frame_range"] == [0, 10]
    assert r["reduction"]["センター"]["input_keys"] == 11
