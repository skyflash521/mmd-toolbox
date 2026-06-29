"""vpr→VMD 変換パイプライン(_convert)の統合テスト(vpr2vmd.md §3〜§5)。

cli._convert は vpr_io 解析結果(合成フィクスチャ)を入口に、トラック選択→重なり解決→口形イベント
確定→開き量→lipsync→モーフキー VMD 出力までを束ねる。vpr_io.read と mmd_toolbox の write_file は
monkeypatch で差し替え、配線と終了コードを決定論的に検証する。
"""

import pytest
from vpr_io import (
    ControllerCurve,
    ControllerEvent,
    Note,
    Part,
    TempoEvent,
    Track,
    VprFormatError,
    VprProject,
)

from mmd_toolbox.vmd import read as vmd_read
from vpr2vmd import cli


def _note(start, dur, phonemes, *, velocity=64):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x",
        velocity=velocity, phonemes=phonemes,
    )


def _project(notes, *, tracks=None, tempos=None, resolution=480):
    part = Part(name="part", start_tick=0, notes=notes)
    track = Track(name="Vocal", parts=[part])
    return VprProject(
        resolution=resolution,
        tempos=tempos if tempos is not None else [TempoEvent(0, 120.0)],
        tracks=tracks if tracks is not None else [track],
    )


def _patch_read(monkeypatch, project):
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))


def _run(monkeypatch, tmp_path, project, *args):
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")  # 存在確認を通す(内容は monkeypatch 済み read が無視)
    out = tmp_path / "out.vmd"
    _patch_read(monkeypatch, project)
    rc = cli.main([str(src), "-o", str(out), *args])
    return rc, out


def _read_doc(path):
    doc, _ = vmd_read(str(path))
    return doc


def _morph_names(path):
    return [k.name for k in _read_doc(path).morph]


def test_convert_single_vowel_writes_morph_vmd(monkeypatch, tmp_path):
    # [a] 音符1つ → モーフキー VMD を出力(あ モーフを含む)。
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]))
    assert rc == 0
    assert out.exists()
    assert "あ" in _morph_names(out)


def test_convert_output_has_frame0_keys_for_used_morphs(monkeypatch, tmp_path):
    # 出力VMDは使用モーフを 0F に中立登録する(編集・MMD互換規約。ensure_frame0_neutral_keys 経由)。
    rc, out = _run(
        monkeypatch, tmp_path, _project([_note(0, 240, ["a"]), _note(480, 240, ["i"])])
    )
    assert rc == 0
    doc = _read_doc(out)
    used = {k.name for k in doc.morph}
    zero = {k.name for k in doc.morph if k.frame == 0}
    assert used  # 使用モーフがある
    assert used <= zero  # 各使用モーフに 0F キーがある


def test_convert_cli_overrides_reach_generation_params(monkeypatch, tmp_path):
    # CLI 調整は最終的な GenerationParams へ届く(指定フィールドを取り違えず上書き)。
    captured = {}

    real = cli.generate_morph_keys

    def spy(events, params):
        captured["params"] = params
        return real(events, params)

    monkeypatch.setattr(cli, "generate_morph_keys", spy)
    rc, _out = _run(
        monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]),
        "--anticipation", "7", "--coartic-overlap", "5",
        "--valley-shallow", "0.5", "--valley-deep", "0.25", "--valley-slope", "0.03",
    )
    assert rc == 0
    p = captured["params"]
    assert p.anticipation_frames == 7
    assert p.coartic_overlap_max == 5
    assert (p.legato_valley_shallow, p.legato_valley_deep, p.legato_valley_slope) == (
        0.5, 0.25, 0.03,
    )


def test_convert_ref_bpm_override_changes_tempo_correction(monkeypatch, tmp_path):
    # --ref-bpm はテンポ補正の入力。基準を代表BPMに合わせると s=1.0 で縮まない。
    captured = {}

    real = cli.generate_morph_keys

    def spy(events, params):
        captured["params"] = params
        return real(events, params)

    monkeypatch.setattr(cli, "generate_morph_keys", spy)
    project = _project([_note(0, 480, ["a"])], tempos=[TempoEvent(0, 190.0)])
    rc, _out = _run(monkeypatch, tmp_path, project, "--ref-bpm", "190")
    assert rc == 0
    # ref-bpm=190・代表BPM=190 → s=1.0。pop の min_hold 基礎値 3 のまま(既定 ref120 なら 2 へ縮む)。
    assert captured["params"].min_hold_frames == 3


def test_convert_loudness_controller_drives_open_amount(monkeypatch, tmp_path):
    # 声量コントローラ(dynamics)があれば velocity でなく曲線から開き量を出す。大音量の音符の開き量が
    # 小音量より大きくなる(velocity は一様でも声量曲線で強弱が出る)。
    captured = {}

    real = cli.build_mouth_events

    def spy(*args, **kwargs):
        captured["open"] = kwargs.get("open_by_note")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "build_mouth_events", spy)
    part = Part(
        name="p",
        start_tick=0,
        notes=[_note(0, 240, ["a"]), _note(480, 240, ["a"])],
        controllers=[
            ControllerCurve(
                name="dynamics",
                events=[ControllerEvent(0, 120), ControllerEvent(480, 10)],
            )
        ],
    )
    project = VprProject(
        resolution=480, tempos=[TempoEvent(0, 120.0)], tracks=[Track(name="Vocal", parts=[part])]
    )
    rc, _out = _run(monkeypatch, tmp_path, project)
    assert rc == 0
    assert captured["open"][0] > captured["open"][1]  # 大音量(note0) > 小音量(note1)


def test_convert_tempo_scale_min_override_reaches_correction(monkeypatch, tmp_path):
    # --tempo-scale-min はテンポ補正の下げ止まり係数として apply_tempo_correction へ届く。
    captured = {}

    real = cli.generate_morph_keys

    def spy(events, params):
        captured["params"] = params
        return real(events, params)

    monkeypatch.setattr(cli, "generate_morph_keys", spy)
    project = _project([_note(0, 480, ["a"])], tempos=[TempoEvent(0, 600.0)])
    rc, _out = _run(monkeypatch, tmp_path, project, "--tempo-scale-min", "0.2")
    assert rc == 0
    # 600bpm・ref120 → 比0.2。s_min=0.2 まで下がり s=0.2、min_hold=round(3*0.2)=1(下限1)。
    # 既定 s_min=0.5 なら s=0.5 で min_hold=2 になるので、上書きが効いていることを固定。
    assert captured["params"].min_hold_frames == 1


def test_convert_valley_deep_above_shallow_is_arg_error(monkeypatch, tmp_path):
    # 谷係数の下限(deep)が上限(shallow)を上回る指定は不正(pop 既定 shallow=0.45 との組み合わせ)。
    rc, _out = _run(
        monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--valley-deep", "0.6"
    )
    assert rc == 2


def test_convert_legato_max_override_reaches_build_mouth_events(monkeypatch, tmp_path):
    # --legato-max は口形イベント確定段(間隙分類)の入力として渡る。
    captured = {}

    real = cli.build_mouth_events

    def spy(*args, **kwargs):
        captured["legato"] = kwargs.get("legato_max_frames")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "build_mouth_events", spy)
    rc, _out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--legato-max", "12")
    assert rc == 0
    assert captured["legato"] == 12.0


def test_convert_writes_only_morph_section(monkeypatch, tmp_path):
    # 生成するのはモーフキーのみ。ボーン・カメラ・照明・セルフ影・IK は空(vpr2vmd.md §5)。
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]))
    assert rc == 0
    doc = _read_doc(out)
    assert doc.morph  # モーフキーは生成される
    assert doc.bone == []
    assert doc.camera == []
    assert doc.light == []
    assert doc.self_shadow == []
    assert doc.ik_property == []


def test_convert_empty_notes_writes_empty_morph_vmd(monkeypatch, tmp_path):
    # 採用音符列が空(発音無し)→ エラーにせず空のモーフキー VMD を出力(コード0)。
    rc, out = _run(monkeypatch, tmp_path, _project([]))
    assert rc == 0
    assert out.exists()
    assert _morph_names(out) == []


def test_convert_no_tracks_is_input_error(monkeypatch, tmp_path):
    # 対象トラックが1件も無い → 入力不正(コード1)。
    rc, _ = _run(monkeypatch, tmp_path, _project([], tracks=[]))
    assert rc == 1


def test_convert_track_index_out_of_range_is_arg_error(monkeypatch, tmp_path):
    # --track INDEX 範囲外 → 引数エラー(コード2)。
    rc, _ = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--track", "5")
    assert rc == 2


def test_convert_track_name_no_match_is_arg_error(monkeypatch, tmp_path):
    # --track NAME 不一致 → 引数エラー(コード2)。
    rc, _ = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]), "--track", "Nope")
    assert rc == 2


def test_convert_vpr_format_error_is_input_error(monkeypatch, tmp_path):
    # vpr_io の読み込み/形式検証失敗(非vpr 等)→ 入力不正(コード1)。
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out = tmp_path / "out.vmd"

    def _raise(_src):
        raise VprFormatError("not a vpr")

    monkeypatch.setattr(cli, "read", _raise)
    assert cli.main([str(src), "-o", str(out)]) == 1


def test_convert_moraic_nasal_uses_n_morph_by_default(monkeypatch, tmp_path):
    # 既定は「ん」モーフを使う。撥音「ん」音符 → ん モーフキーを含む。
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["N\\"])]))
    assert rc == 0
    assert "ん" in _morph_names(out)


def test_convert_no_n_morph_drops_n_morph(monkeypatch, tmp_path):
    # --no-n-morph 指定時、単独撥音は無音(閉口)へ倒れモーフキーを一切出さない
    # (「ん」不在に加え、誤って「あ」等へ倒さないことも固定)。
    rc, out = _run(
        monkeypatch, tmp_path, _project([_note(0, 480, ["N\\"])]), "--no-n-morph"
    )
    assert rc == 0
    assert _morph_names(out) == []


def test_convert_write_failure_is_output_error(monkeypatch, tmp_path):
    # 出力 VMD の書き込み失敗 → 出力書き込み失敗(コード3)。
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    out = tmp_path / "out.vmd"
    _patch_read(monkeypatch, _project([_note(0, 480, ["a"])]))

    def _raise(_doc, _path):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_file", _raise)
    assert cli.main([str(src), "-o", str(out)]) == 3
