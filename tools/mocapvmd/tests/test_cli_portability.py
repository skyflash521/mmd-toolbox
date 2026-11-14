from mocapvmd import cli
from vmd import io as vmd_io

from .helpers import bone, write_vmd


def _ramp(path):
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(11)])


def test_machine_stdout_lines_are_nonempty_lf_terminated_without_cr(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _ramp(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    for line in raw.split(b"\n")[:-1]:
        assert line and not line.endswith(b"\r")


def test_machine_stdout_writes_non_ascii_output_path_as_raw_utf8(tmp_path, capsysbinary):
    src = tmp_path / "入力.vmd"
    _ramp(src)
    out = tmp_path / "出力.vmd"
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert "出力".encode("utf-8") in raw
    text = raw.decode("utf-8")
    assert "出力" in text


def test_non_ascii_input_and_output_paths_produce_readable_vmd(tmp_path):
    src = tmp_path / "モーション入力.vmd"
    _ramp(src)
    out = tmp_path / "モーション出力.vmd"
    rc = cli.main([str(src), "-o", str(out), "--no-reduce"])
    assert rc == 0
    assert out.exists()
    doc, _ = vmd_io.read(str(out))
    assert doc.bone
