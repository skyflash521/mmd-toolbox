from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


def _mmd_file(name: str) -> bytes:
    path = DATA_DIR / name
    if not path.exists():
        pytest.fail(
            f"MMD産テストデータがありません: {path}\n"
            "テストデータを作成してください"
        )
    return path.read_bytes()


@pytest.fixture
def camera_basic_bytes() -> bytes:
    return _mmd_file("camera_basic.vmd")


@pytest.fixture
def model_motion_bytes() -> bytes:
    return _mmd_file("model_motion.vmd")
