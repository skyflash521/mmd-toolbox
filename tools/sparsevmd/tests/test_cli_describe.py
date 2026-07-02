"""sparsevmd CLI 自己記述 --describe のテスト(sparsevmd.md §12.3)。

--describe は VMD を読まず、オプション定義とプリセット一覧の result イベント(mode:"describe")を
出して終了する独立メタ操作。--machine を要さず単独で起動でき、入力 positional も要求しない。
options は処理を駆動する引数の配列で、各要素は {name, type, constraint, default, help, repeat} の
6 キー。真偽フラグの否定形(--no-cut-detect)とメタ/モード操作(--describe/--version/--help/--machine)は
options に載せない。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。テスト方針は
../../../libs/vmd/vmd.md §4 に準ずる(決定論的・外部依存なし)。
"""

import json

from sparsevmd import cli, presets

_GROUPS = ["core", "arms", "legs", "fingers", "ik", "mocap"]
_NN = {"min": 0, "max": None, "exclusive_min": False}
_FOV = {"min": 0.5, "max": None, "exclusive_min": False}
_INT1 = {"min": 1, "max": None, "exclusive_min": False}
_FRAME = {"min": 0, "max": None, "exclusive_min": False}


def _num_field(name, type_, mn):
    return {"name": name, "type": type_, "min": mn, "max": None, "exclusive_min": False}


# §12.3 の 31 行表。name → (type, constraint, default, repeat)。JSON 経由で tuple は list、None は null。
EXPECTED = {
    "input": ("str", None, None, False),
    "--output": ("str", None, None, False),
    "--overwrite": ("flag", None, False, False),
    "--target": ("enum", {"choices": ["camera", "bone", "all"]}, "all", False),
    "--bone": ("str", None, None, True),
    "--bone-glob": ("str", None, None, True),
    "--bone-group": ("enum", {"choices": _GROUPS}, None, True),
    "--bone-file": ("str", None, None, False),
    "--exclude-bone": ("str", None, None, True),
    "--exclude-bone-glob": ("str", None, None, True),
    "--exclude-bone-group": ("enum", {"choices": _GROUPS}, None, True),
    "--list-bones": ("flag", None, False, False),
    "--range": ("compound",
                {"format": "START:END",
                 "fields": [_num_field("START", "int", 0), _num_field("END", "int", 0)]},
                None, True),
    "--preset": ("enum", {"choices": list(presets.PRESET_NAMES)}, "balanced", False),
    "--bone-pos-tol": ("float", _NN, None, False),
    "--bone-rot-tol": ("float", _NN, None, False),
    "--camera-pos-tol": ("float", _NN, None, False),
    "--camera-rot-tol": ("float", _NN, None, False),
    "--camera-distance-tol": ("float", _NN, None, False),
    "--camera-fov-tol": ("float", _FOV, None, False),
    "--max-segment-frames": ("int", _INT1, None, False),
    "--min-segment-frames": ("int", _INT1, 1, False),
    "--curve-mode": ("enum", {"choices": ["bezier", "linear"]}, "bezier", False),
    "--strict": ("flag", None, False, False),
    "--cut-threshold-camera": ("compound",
                               {"format": "POS,ROT,DIST",
                                "fields": [_num_field("POS", "float", 0), _num_field("ROT", "float", 0),
                                           _num_field("DIST", "float", 0)]},
                               [5.0, 20.0, 5.0], False),
    "--cut-threshold-bone": ("compound",
                             {"format": "POS,ROT",
                              "fields": [_num_field("POS", "float", 0), _num_field("ROT", "float", 0)]},
                             [1.0, 30.0], False),
    "--cut-detect": ("flag", None, True, False),
    "--keep-frame": ("int", _FRAME, None, True),
    "--dry-run": ("flag", None, False, False),
    "--verbose": ("flag", None, False, False),
    "--quiet": ("flag", None, False, False),
}


def describe_result(capsysbinary):
    """--describe の stdout を解析し、単一の result(mode:"describe")イベントを返す。"""
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_emits_result_without_input(capsysbinary):
    # --describe は入力を要求せず、VMD を読まずに options/presets の result を出して exit 0。
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list)
    # describe の result は options/presets のみ(§12.3)。reduce/inspect/list_bones のキーは載せない。
    assert set(r) == {"type", "mode", "options", "presets"}


def test_describe_works_without_machine_flag(capsysbinary):
    # --describe は --machine を要さない独立メタ操作(--machine 無しでも構造化 result を出す)。
    rc = cli.main(["--describe"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_ignores_input_and_does_not_read_vmd(capsysbinary):
    # --describe は VMD を読まない独立メタ操作(§12.3)。存在しない入力パスを渡しても、パス検証・
    # 読み込みをせず describe result を出して exit 0(入力 positional に阻まれない)。
    rc = cli.main(["--describe", "does_not_exist.vmd"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_shape_and_repeat(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    # メタ/モード操作は options に含めない。
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    # 否定形は重複列挙しない(肯定形の長形式のみ)。
    assert "--no-cut-detect" not in by_name
    # 全 31 要素・各要素は常に 6 キー・help は非空文字列。
    assert len(r["options"]) == 31
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help", "repeat"}
        assert isinstance(o["help"], str) and o["help"]
        assert isinstance(o["repeat"], bool)
    # §12.3 の 31 行表を全要素の {type, constraint, default, repeat} で固定する。
    assert set(by_name) == set(EXPECTED)
    for name, (type_, constraint, default, repeat) in EXPECTED.items():
        o = by_name[name]
        assert o["type"] == type_, name
        assert o["constraint"] == constraint, name
        assert o["default"] == default, name
        assert o["repeat"] == repeat, name


def test_describe_presets_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    for p in r["presets"]:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {"bone_pos_tol", "bone_rot_tol", "camera_pos_tol",
                                    "camera_rot_tol", "camera_distance_tol", "camera_fov_tol"}
    # 3 プリセットの §2.4 の値を全件固定する。
    expected = {
        "precise": {"bone_pos_tol": 0.005, "bone_rot_tol": 0.05, "camera_pos_tol": 0.01,
                    "camera_rot_tol": 0.02, "camera_distance_tol": 0.01, "camera_fov_tol": 0.50},
        "balanced": {"bone_pos_tol": 0.01, "bone_rot_tol": 0.10, "camera_pos_tol": 0.02,
                     "camera_rot_tol": 0.05, "camera_distance_tol": 0.02, "camera_fov_tol": 0.50},
        "aggressive": {"bone_pos_tol": 0.05, "bone_rot_tol": 0.50, "camera_pos_tol": 0.10,
                       "camera_rot_tol": 0.25, "camera_distance_tol": 0.10, "camera_fov_tol": 1.00},
    }
    assert {p["name"]: p["values"] for p in r["presets"]} == expected


def test_describe_type_table_covers_non_meta_args():
    # _D_TYPE はメタ/モード操作を除く全 parser 引数を覆う。parser に引数を足して _D_TYPE への追加を
    # 忘れると describe から黙って抜けるため、その載せ忘れをここで検出する。
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_mode_arg_error_is_error_event(capsysbinary):
    # --describe(--machine 無し)も構造化出力モードなので、引数エラーは標準エラーでなく error
    # イベントでストリームを終端する(§12.1)。
    rc = cli.main(["--describe", "--max-segment-frames", "abc"])
    assert rc == 2
    out = capsysbinary.readouterr().out
    events = [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "bad_argument"
    assert events[-1]["field"] == "--max-segment-frames" and events[-1]["exit_code"] == 2
