"""vpr→VMD 変換パイプライン(cli._build と main)の統合テスト。

cli._build は vpr 解析結果(合成フィクスチャ)を入口に、トラック選択→重なり解決→口形イベント
確定→開き量→lipsync→モーフキー生成までを束ね、main が書き込み・診断表示・警告を担う。
vpr.read と vmd の write_file は monkeypatch で差し替え、配線と終了コードを決定論的に
検証する。
"""

import re

import pytest
from vpr import (
    ControllerCurve,
    ControllerEvent,
    Note,
    Part,
    TempoEvent,
    Track,
    VprFormatError,
    VprProject,
)

from vmd import read as vmd_read
from vpr2vmd import __version__, cli


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


@pytest.mark.xfail(reason="impl pending: vpr2vmd --model-name default", strict=True)
def test_convert_default_model_name_is_tool_and_version(monkeypatch, tmp_path):
    # --model-name 未指定時、出力VMDの model_name はツール名+実行中のバージョン。
    rc, out = _run(monkeypatch, tmp_path, _project([_note(0, 480, ["a"])]))
    assert rc == 0
    assert _read_doc(out).model_name == f"vpr2vmd {__version__}"


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
    # 生成するのはモーフキーのみ。ボーン・カメラ・照明・セルフ影・IK は空。
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
    # vpr の読み込み/形式検証失敗(非vpr 等)→ 入力不正(コード1)。
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


# --- 診断(--dry-run) ---


def _dry_run(monkeypatch, tmp_path, project, capsys, *args):
    """read を差し替えて --dry-run を走らせ、(rc, stdout, stderr) を返す。"""
    src = tmp_path / "in.vpr"
    src.write_bytes(b"")
    _patch_read(monkeypatch, project)
    rc = cli.main([str(src), "--dry-run", *args])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_dry_run_reports_adopted_event_and_morph_counts(monkeypatch, tmp_path, capsys):
    # --dry-run は採用音符数・口形イベント数・モーフキー数を診断に出す。
    project = _project([_note(0, 240, ["a"]), _note(480, 240, ["i"])])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "採用音符数: 2" in out
    # ラベルだけでなく件数値(正の整数)を固定する(空値・見出しのみの実装を弾く)。
    ev = re.search(r"口形イベント数:\s*(\d+)", out)
    assert ev and int(ev.group(1)) >= 1
    mk = re.search(r"モーフキー数:\s*(\d+)", out)
    assert mk and int(mk.group(1)) >= 1


def test_dry_run_reports_openness_stats(monkeypatch, tmp_path, capsys):
    # 開き量統計(最小/最大/平均)を出す(velocity に強弱差のある2音符)。
    project = _project(
        [_note(0, 240, ["a"], velocity=40), _note(480, 240, ["i"], velocity=120)]
    )
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    # 見出しだけでなく最小/最大/平均の3値を固定する。velocity 差があるので最小<最大。
    m = re.search(r"開き量[^\n]*?([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", out)
    assert m, "開き量の最小/最大/平均が出ていない"
    lo, hi, avg = (float(m.group(i)) for i in (1, 2, 3))
    assert lo < hi
    assert lo <= avg <= hi


def test_dry_run_reports_vowel_undetermined_count(monkeypatch, tmp_path, capsys):
    # 母音が得られない音符(母音なし・撥音/促音でもない)を母音未確定として計上する。
    # 母音音符に続けて、その他子音のみの音符(直前口形継続=母音未確定)を置く。
    project = _project([_note(0, 240, ["a"]), _note(480, 240, ["k"])])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "母音未確定: 1" in out


def test_dry_run_reports_overlap_exclusion_and_truncation(monkeypatch, tmp_path, capsys):
    # 同一 start の重複は除外、後続開始への切り詰めは切り詰めとして計上する。
    project = _project([
        _note(0, 480, ["a"]),
        _note(0, 240, ["i"]),    # 同一 start → 除外1件
        _note(480, 480, ["u"]),  # 終端960が次音符start720を越える → 切り詰め1件
        _note(720, 240, ["e"]),
    ])
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "除外: 1" in out
    assert "切り詰め: 1" in out


def test_dry_run_lists_non_event_symbols(monkeypatch, tmp_path, capsys):
    # 自前の口形イベントを作らない記号(その他子音・未知記号)を記号種・件数で列挙する。
    # 表明はラベルと「記号(件数)」形でパス文字列への偶発一致を避ける(単独 "k" 等は不可)。
    project = _project([_note(0, 480, ["k", "a"])])  # k は OTHER(自前イベントを作らない)
    rc, out, _err = _dry_run(monkeypatch, tmp_path, project, capsys)
    assert rc == 0
    assert "イベント外記号" in out
    assert "k(1)" in out


def test_dry_run_empty_track_warns_on_stderr(monkeypatch, tmp_path, capsys):
    # 採用音符列が空 → 標準エラーへ警告を出す(--dry-run でも、出力VMDは書かない・exit 0)。
    rc, _out, err = _dry_run(monkeypatch, tmp_path, _project([]), capsys)
    assert rc == 0
    assert "warning: no_adopted_notes: " in err


def test_empty_track_warns_on_stderr_in_normal_run(monkeypatch, tmp_path, capsys):
    # 通常実行でも採用音符列が空なら標準エラーへ警告を出す(空VMD出力・exit 0)。
    rc, out = _run(monkeypatch, tmp_path, _project([]))
    assert rc == 0
    assert out.exists()
    assert "warning: no_adopted_notes: " in capsys.readouterr().err
