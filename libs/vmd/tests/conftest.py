from pathlib import Path

import pytest

from vmd import read

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_CAMERA_INTERP = bytes([20, 107, 20, 107]) * 6


def _mmd_file(name: str) -> bytes:
    path = DATA_DIR / name
    if not path.exists():
        pytest.fail(
            f"MMD産テストデータがありません: {path}\n"
            "テストデータを作成してください"
        )
    return path.read_bytes()


def _require(name: str, unmet: list[str]) -> None:
    if unmet:
        pytest.fail(
            f"テストデータの作り直しが必要: {name}\n" + "\n".join(f"- {u}" for u in unmet)
        )


def _camera_basic_unmet(doc) -> list[str]:
    cams = doc.camera
    later = [k.interpolation for k in cams[1:]]
    conditions = [
        (len(cams) >= 3, "カメラキーが3個以上必要"),
        (any(i == DEFAULT_CAMERA_INTERP for i in later), "補間曲線がデフォルト値のキー(2個目以降)が1個以上必要"),
        (any(i != DEFAULT_CAMERA_INTERP for i in later), "補間曲線がデフォルト値でないキー(2個目以降)が1個以上必要"),
        (len({k.distance for k in cams}) > 1, "距離の値が2種類以上必要"),
        (len({k.position for k in cams}) > 1, "カメラ中心の座標(X・Y・Zの組)が2種類以上必要"),
        (len({k.rotation for k in cams}) > 1, "角度(X・Y・Zの組)が2種類以上必要"),
        (len({k.fov for k in cams}) > 1, "視野角の値が2種類以上必要"),
        ({k.perspective for k in cams} == {0, 1}, "パースONのキーとOFFのキーの両方が必要"),
        (len(doc.light) >= 1, "照明キーが1個以上必要"),
        (len(doc.self_shadow) >= 1, "セルフ影キーが1個以上必要"),
    ]
    return [msg for ok, msg in conditions if not ok]


def _model_motion_unmet(doc) -> list[str]:
    conditions = [
        (len(doc.bone) >= 1, "ボーンキーが1個以上必要"),
        (len(doc.morph) >= 1, "モーフキーが1個以上必要"),
    ]
    return [msg for ok, msg in conditions if not ok]


@pytest.fixture
def camera_basic_bytes() -> bytes:
    data = _mmd_file("camera_basic.vmd")
    _require("camera_basic.vmd", _camera_basic_unmet(read(data)[0]))
    return data


@pytest.fixture
def model_motion_bytes() -> bytes:
    data = _mmd_file("model_motion.vmd")
    _require("model_motion.vmd", _model_motion_unmet(read(data)[0]))
    return data
