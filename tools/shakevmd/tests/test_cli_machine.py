"""shakevmd CLI 機械モードのテスト。

機械モード(--machine)を検証する: stdout を JSON Lines のイベント専用にし、result/warning/progress
イベントを出す。ストリームは result または error のちょうど 1 つで終端する(本モジュールは成功=result
終端を対象にし、error イベントは構造化エラーのテストで扱う)。非機械モードの既定挙動が不変であること
(後方互換)も併せて検証する。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
テストは決定論的・外部依存なしで行う。
"""

import json

from shakevmd import bake as bake_mod
from shakevmd import cli, presets
from vmd import io
from vmd.types import BoneKey, CameraKey, MorphKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0)),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05)),
]


def write_input(path, keys=KEYS, **doc_kwargs):
    io.write_file(VmdDocument(camera=list(keys), **doc_kwargs), str(path))
    return str(path)


def machine_events(capsysbinary):
    """capsysbinary で捕捉した stdout を JSON Lines として解析しイベント配列で返す。"""
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")  # UTF-8 固定(ロケール非依存)を前提に decode
    return [json.loads(ln) for ln in text.split("\n") if ln]


def machine_error(capsysbinary):
    """機械モードの stdout を解析し、終端の error イベントを返す(失敗は error で終端)。"""
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど1つ
    return events[-1]


def test_machine_emits_result_event(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    # ちょうど 1 つの終端(result)で終わる。
    assert events[-1]["type"] == "result"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    result = events[-1]
    assert result["mode"] == "bake"
    assert result["output"] == str(out)
    assert result["keys"] == 61  # --no-smooth の密キー(0..60)
    assert isinstance(result["applied_ranges"], list)
    assert isinstance(result["max_amplitude"], float)
    assert isinstance(result["detected_cuts"], list)


def test_machine_stdout_is_valid_json_lines(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    objs = []
    for ln in text.split("\n")[:-1]:
        assert ln != ""  # 空行を挟まない
        obj = json.loads(ln)  # 各行が単一 JSON
        assert "type" in obj
        objs.append(obj)
    assert objs[-1]["type"] == "result" and objs[-1]["mode"] == "bake"  # 成功は result(bake)で終端


def test_machine_no_human_text_on_stdout(tmp_path, capsysbinary):
    # 機械モードの stdout は人間向けテキスト(range:/keys:/warning: 等)を含まない(チャネル固定)。
    inp = write_input(tmp_path / "in.vmd")
    cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--verbose", "--no-smooth"])
    text = capsysbinary.readouterr().out.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert lines  # 機械モードは少なくとも result を出す(stdout が空でない)
    for ln in lines:
        json.loads(ln)  # すべて JSON、人間向けテキスト行は混入しない


def test_machine_emits_warning_event_for_non_camera_sections(tmp_path, capsysbinary):
    # カメラ以外のセクションを含む入力 → non_camera_sections_passthrough の warning イベント。
    # 非カメラセクションを 2 種(bone と morph)含め、section 配列が全セクション名を載せることを確認。
    bone = [BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                    position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                    interpolation=bytes(64))]
    morph = [MorphKey(name_raw=b"morph".ljust(15, b"\x00"), frame=0, weight=0.0)]
    inp = write_input(tmp_path / "in.vmd", bone=bone, morph=morph)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    warns = [e for e in events if e["type"] == "warning"]
    codes = {w["code"] for w in warns}
    assert "non_camera_sections_passthrough" in codes
    w = next(w for w in warns if w["code"] == "non_camera_sections_passthrough")
    # section は透過した全セクション名の配列。bone と morph の両方を載せる。
    assert isinstance(w["section"], list)
    assert set(w["section"]) == {"bone", "morph"}
    assert isinstance(w["message"], str) and w["message"]  # 自由文字列の文言を message に保持


def test_machine_emits_warning_event_for_duplicate_frame(tmp_path, capsysbinary):
    # 同一フレーム重複の後勝ち破棄 → bake_normalize_duplicate の warning イベント(section=["camera"])。
    dup = [cam(0), cam(30), cam(30, center=(9.0, 9.0, 9.0)), cam(60)]
    inp = write_input(tmp_path / "in.vmd", keys=dup)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "bake_normalize_duplicate")
    assert w["section"] == ["camera"]
    assert isinstance(w["message"], str) and w["message"]


def test_machine_emits_warning_event_for_octave_clamp(tmp_path, capsysbinary):
    # 実効周波数が帯域上限を超えるオクターブのクランプ → octave_clamped の warning イベント
    # (section=null)。内蔵 octaves=3 では freq×4>8(=freq>2)でクランプが起きる。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth", "--freq", "3.0"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "octave_clamped")
    assert w["section"] is None
    assert isinstance(w["message"], str) and w["message"]


def test_machine_emits_warning_event_for_fade_shortened(tmp_path, capsysbinary):
    # 範囲長が 2×fade 未満でのフェード自動短縮 → fade_shortened の warning イベント(section=null)。
    # 既定 fade=0.7 → 2×fade=42 フレーム。範囲[0,30]=31 フレーム<42 で短縮が起きる。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth", "--range", "0:30"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "fade_shortened")
    assert w["section"] is None
    assert isinstance(w["message"], str) and w["message"]


def test_machine_emits_progress_events(tmp_path, capsysbinary):
    # 機械モードでは進捗をイベントとして出す(TTY 判定に依存しない)。ベイク段の progress を含む。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    progress = [e for e in events if e["type"] == "progress"]
    assert progress, "progress イベントが少なくとも 1 本出る"
    stages = {p["stage"] for p in progress}
    assert "bake" in stages
    for p in progress:
        assert set(p) >= {"type", "stage", "done", "total", "note", "elapsed"}


def test_machine_smooth_emits_smooth_progress(tmp_path, capsysbinary):
    # 既定 on のスムージング段も機械モードで progress イベントを出す。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    stages = {e["stage"] for e in events if e["type"] == "progress"}
    assert "smooth" in stages


def test_non_machine_default_unchanged(tmp_path, capsys):
    # 後方互換: --machine なしの既定挙動(出力 VMD・終了コード)は不変。stdout に JSON を出さない。
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--no-smooth"])
    assert rc == 0
    assert out.exists()
    stdout = capsys.readouterr().out
    # 既定実行は dry-run/verbose でない限り統計を出さない(JSON も出さない)。
    assert stdout.strip() == "" or not stdout.lstrip().startswith("{")


def test_machine_version_help_stay_human(capsys):
    # --machine 併用でも --version/--help は人間向けテキストを出して exit 0 で終わり、
    # イベントストリームには載せない(メタ操作の例外)。argparse が両者を
    # 処理面より先に短絡するため --machine 追加の前後で不変であることを保証する。
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "shakevmd" in out and not out.lstrip().startswith("{")  # 人間向け、JSON でない
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_machine_flag(capsys):
    # --help は人間向けテキストを出して exit 0(main は argparse の SystemExit を握って 0 を返す)。
    # 新設の --machine がヘルプに現れること(人間向けヘルプ)を確認する。
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "--machine" in text


# --- 構造化エラー -----------------------------------------------
# 各失敗経路が機械モードで確定 code/field/exit_code の error イベントを出してストリームを終端し、
# 終了コードを維持することを検証する。非機械モードは理由を標準エラーへ1行出す。


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    # 未知オプション → argparse 検出の bad_argument(exit 2)。error で終端する。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--bogus"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    # positional input 欠落 → bad_argument、field は input。
    rc = cli.main(["--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input"


def test_machine_error_bad_argument_invalid_value_field(tmp_path, capsysbinary):
    # 型エラー(--seed 非整数)→ bad_argument、field は該当オプションの長形式。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "--machine", "--seed", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--seed"


def test_machine_error_not_vmd(tmp_path, capsysbinary):
    # 非VMD/破損入力 → not_vmd(exit 1)、field は input。握り潰していた例外種別を message に載せる。
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_vmd" and e["field"] == "input" and e["exit_code"] == 1
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_no_camera_keys(tmp_path, capsysbinary):
    # カメラキー0件 → no_camera_keys(exit 1)。
    p = str(tmp_path / "nocam.vmd")
    io.write_file(VmdDocument(camera=[]), p)
    rc = cli.main([p, "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "no_camera_keys" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_output_overwrites_input(tmp_path, capsysbinary):
    # 出力が入力と同一パス・--overwrite 未指定 → output_overwrites_input(exit 2)、field は --output。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", inp, "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_overwrites_input" and e["field"] == "--output"


def test_machine_error_range_reversed(tmp_path, capsysbinary):
    # 省略端の解決後に逆順(999: の END=末尾<999)→ range_reversed(exit 2)、field は --range。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--range", "999:"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "range_reversed" and e["field"] == "--range"


def test_machine_error_range_overlap(tmp_path, capsysbinary):
    # 範囲の重複/接触 → bake の ValueError → range_overlap(exit 2)、field は --range。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine",
                   "--range", "0:30", "--range", "30:60"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "range_overlap" and e["field"] == "--range"


def test_machine_error_value_overflow(tmp_path, capsysbinary):
    # 過大値がベイク中に float32 で溢れる(--fade 1e308)→ value_overflow(exit 2)、field は null。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--fade", "1e308"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "value_overflow" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_non_finite_output(tmp_path, capsysbinary, monkeypatch):
    # ベイクが非有限(inf/nan)の出力を返した場合 → non_finite_output(exit 2)、field は null。
    # 通常の CLI 引数(有限)ではベイクが例外側に倒れて到達しにくい防御経路なので、bake を差し替えて
    # 有限性検査(_all_finite)の分岐を直接検証する。
    inf_key = CameraKey(0, -30.0, (float("inf"), 0.0, 0.0), (0.0, 0.0, 0.0), LINEAR, 30, 0)

    def bad_bake(camera_keys, *a, **k):
        return bake_mod.BakeResult(camera_keys=[inf_key], warnings=[], resolved=[(0, 0)])

    monkeypatch.setattr(cli, "bake", bad_bake)
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "non_finite_output" and e["field"] is None


def test_machine_error_write_failed(tmp_path, capsysbinary):
    # 出力の I/O 失敗(親がファイル)→ write_failed(exit 3)、field は --output、path 付き。
    inp = write_input(tmp_path / "in.vmd")
    clash = tmp_path / "afile"
    clash.write_bytes(b"x")
    out = str(clash / "out.vmd")
    rc = cli.main([inp, "-o", out, "--machine", "--no-smooth"])
    assert rc == 3
    e = machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_internal_error(tmp_path, capsysbinary, monkeypatch):
    # 想定外の内部例外(bake が RuntimeError)→ internal_error(exit 1)。安全網。
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cli, "bake", boom)
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["exit_code"] == 1


def test_non_machine_error_prints_reason_to_stderr(tmp_path, capsys):
    # 非機械モードでも失敗理由を標準エラーへ1行出す。終了コードは維持し、stdout に JSON は出さない。
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad)])
    assert rc == 1
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


# --- 自己記述 --describe ----------------------------------------


def describe_result(capsysbinary):
    """--describe の stdout を解析し、単一の result(mode:"describe")イベントを返す。"""
    events = machine_events(capsysbinary)
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_emits_result_without_input(capsysbinary):
    # --describe は入力を要求せず、VMD を読まずに options/presets の result を出して exit 0。
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list)
    # bake/inspect 統計キーは describe には載せない。
    for k in ("output", "keys", "applied_ranges", "max_amplitude", "detected_cuts"):
        assert k not in r


def test_describe_works_without_machine_flag(capsysbinary):
    # --describe は --machine を要さない独立メタ操作(--machine 無しでも構造化 result を出す)。
    rc = cli.main(["--describe"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    # メタ/モード操作は options に含めない。
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    # 各要素は常に5キー、help は非空文字列。
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert isinstance(o["help"], str) and o["help"]
    # positional input。
    assert by_name["input"]["type"] == "str" and by_name["input"]["constraint"] is None
    # 揺れ float: 非負制約・解決後の hard-default。
    assert by_name["--amp-rot"]["type"] == "float"
    assert by_name["--amp-rot"]["constraint"] == {"min": 0, "max": None, "exclusive_min": False}
    assert by_name["--amp-rot"]["default"] == 0.8
    # --freq は正(exclusive_min=true)。
    assert by_name["--freq"]["constraint"]["exclusive_min"] is True
    # 裸の int(--seed)は constraint null・default 1。
    assert by_name["--seed"]["type"] == "int" and by_name["--seed"]["constraint"] is None
    assert by_name["--seed"]["default"] == 1
    # enum(--preset)。
    assert by_name["--preset"]["type"] == "enum"
    assert set(by_name["--preset"]["constraint"]["choices"]) == set(presets.PRESET_NAMES)
    assert by_name["--preset"]["default"] is None
    # flag: --smooth は既定 on=true、store_true 系は false。名前は否定形でない長形式。
    assert by_name["--smooth"]["type"] == "flag" and by_name["--smooth"]["default"] is True
    assert by_name["--overwrite"]["type"] == "flag" and by_name["--overwrite"]["default"] is False
    # compound(--rot-weights)。tuple の hard-default は配列化。
    rw = by_name["--rot-weights"]
    assert rw["type"] == "compound" and rw["constraint"]["format"] == "P,Y,R"
    assert [f["name"] for f in rw["constraint"]["fields"]] == ["P", "Y", "R"]
    assert rw["default"] == [1.0, 1.0, 0.3]
    # --output/複数指定系は既定 null。
    assert by_name["--output"]["default"] is None
    assert by_name["--range"]["default"] is None and by_name["--impulse"]["default"] is None


def test_describe_presets_shape(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {p["name"]: p for p in r["presets"]}
    assert set(by_name) == set(presets.PRESET_NAMES)
    for p in r["presets"]:
        assert set(p) == {"name", "values"} and isinstance(p["values"], dict)
        # 内蔵パラメーターは values に出さない。
        for internal in presets.INTERNAL_PARAM_NAMES:
            assert internal not in p["values"]


def test_describe_type_table_covers_non_meta_args():
    # _D_TYPE はメタ/モード操作を除く全 parser 引数を覆う。parser に引数を足して _D_TYPE への
    # 追加を忘れると describe から黙って抜けるため、その載せ忘れをここで検出する。
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_mode_arg_error_is_error_event(capsysbinary):
    # --describe(--machine 無し)も構造化出力モードなので、引数エラーは標準エラーでなく error
    # イベントでストリームを終端する。
    rc = cli.main(["--describe", "--seed", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--seed" and e["exit_code"] == 2


# --- 入力検査 --machine --dry-run(mode:"inspect") -----------------


def test_machine_dry_run_emits_inspect_result(tmp_path, capsysbinary):
    # --machine --dry-run は VMD を書かず、入力メタ情報 + 揺れプレビュー統計の inspect result を出す。
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--dry-run", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = events[-1]
    assert r["type"] == "result" and r["mode"] == "inspect"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど1つ
    assert r["output"] is None and not out.exists()          # 書かない
    assert r["input_kind"] == "camera"
    assert r["keys"] == 3                                     # 入力カメラキー数(0/30/60)。密キーではない
    assert r["frame_range"] == [0, 60]
    assert r["duration_sec"] == 60 / 30.0
    assert isinstance(r["sections"], list) and "camera" in r["sections"]
    assert isinstance(r["applied_ranges"], list)
    assert isinstance(r["max_amplitude"], float)
    assert isinstance(r["detected_cuts"], list)


def test_machine_dry_run_inspect_lists_non_camera_sections(tmp_path, capsysbinary):
    # inspect の sections は camera と混在する非カメラセクションを載せる。
    bone = [BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                    position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                    interpolation=bytes(64))]
    inp = write_input(tmp_path / "in.vmd", bone=bone)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--dry-run", "--no-smooth"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "inspect"
    assert set(r["sections"]) == {"camera", "bone"}


def test_non_machine_dry_run_unchanged(tmp_path, capsys):
    # 非機械の --dry-run は従来どおり VMD を書かず統計を表示し、stdout に JSON は出さない(後方互換)。
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()
    cap = capsys.readouterr()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


# --- 中断 -----------------------------------------------------------


def _raise_keyboard_interrupt(*a, **k):
    raise KeyboardInterrupt()


def test_machine_cancelled_on_keyboard_interrupt(tmp_path, capsysbinary, monkeypatch):
    # 計算中の KeyboardInterrupt → cancelled の error イベント・exit 130。出力は書かれない(原子性)。
    monkeypatch.setattr(cli, "bake", _raise_keyboard_interrupt)
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--no-smooth"])
    assert rc == 130
    e = machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130 and e["field"] is None
    assert not out.exists()


def test_non_machine_cancelled_on_keyboard_interrupt(tmp_path, capsys, monkeypatch):
    # 非機械モードの中断は stdout に JSON を出さず理由を標準エラーへ1行、exit 130。出力は書かれない。
    monkeypatch.setattr(cli, "bake", _raise_keyboard_interrupt)
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--no-smooth"])
    assert rc == 130
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")
    assert not out.exists()
