import io
import json
import zipfile

from vmd import read as vmd_read
from vpr2vmd import cli

_SINGING_TRACK_TYPE = 2


def _write_vpr(path, notes):
    sequence = {
        "version": {"major": 6, "minor": 5, "revision": 1},
        "vender": "Yamaha Corporation",
        "title": "test",
        "masterTrack": {
            "samplingRate": 44100,
            "tempo": {"events": [{"pos": 0, "value": 12000}]},
            "timeSig": {"events": [{"bar": 0, "numer": 4, "denom": 4}]},
        },
        "voices": [],
        "tracks": [{
            "type": _SINGING_TRACK_TYPE,
            "name": "Vocal",
            "parts": [{"name": "part", "pos": 0, "duration": 1920, "notes": notes}],
        }],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Project/sequence.json", json.dumps(sequence, ensure_ascii=False))
    path.write_bytes(buf.getvalue())


def _note(pos, duration, lyric, phoneme):
    return {
        "pos": pos, "duration": duration, "number": 60,
        "lyric": lyric, "phoneme": phoneme, "velocity": 64,
    }


def test_converts_vpr_file_to_readable_morph_vmd(tmp_path):
    src = tmp_path / "in.vpr"
    out = tmp_path / "out.vmd"
    _write_vpr(src, [
        _note(0, 480, "あ", "a"),
        _note(480, 480, "ら", "4 a"),
        _note(960, 480, "い", "i"),
    ])

    assert cli.main([str(src), "-o", str(out)]) == 0

    doc, _ = vmd_read(str(out))
    assert len(doc.morph) > 0


def test_normal_conversion_writes_nothing_to_stderr(tmp_path, capsys):
    src = tmp_path / "in.vpr"
    out = tmp_path / "out.vmd"
    _write_vpr(src, [
        _note(0, 480, "あ", "a"),
        _note(480, 480, "い", "i"),
        _note(960, 480, "う", "u"),
        _note(1440, 480, "ま", "m a"),
    ])

    assert cli.main([str(src), "-o", str(out)]) == 0
    assert capsys.readouterr().err == ""


def test_non_vpr_bytes_report_input_error(tmp_path, capsys):
    src = tmp_path / "in.vpr"
    src.write_bytes(b"this is not a zip archive")
    out = tmp_path / "out.vmd"

    assert cli.main([str(src), "-o", str(out)]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err
    assert not out.exists()


def test_vpr_content_is_accepted_regardless_of_extension(tmp_path):
    src = tmp_path / "in.zip"
    out = tmp_path / "out.vmd"
    _write_vpr(src, [_note(0, 480, "あ", "a")])

    assert cli.main([str(src), "-o", str(out)]) == 0
    assert out.exists()
