from sparsevmd import cli
from vmd import io
from vmd.types import BoneKey, VmdDocument

_BONE_LINEAR = bytes([20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107]) + bytes(48)


def _bone(name, frame):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, (0.0, float(frame), 0.0),
                   (0.0, 0.0, 0.0, 1.0), _BONE_LINEAR)


def test_text_dry_run_lists_bones_in_sorted_name_order_whatever_the_input_order(tmp_path, capsys):
    src = tmp_path / "in.vmd"
    keys = [_bone(name, f) for name in ("B", "A") for f in range(11)]
    io.write_file(VmdDocument(bone=keys), str(src))
    assert cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--dry-run"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("bone ")]
    assert [ln.split(":")[0] for ln in lines] == ["bone A", "bone B"]
