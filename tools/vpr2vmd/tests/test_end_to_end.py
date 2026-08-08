"""vpr ファイルを入力に CLI を通し、書き出した VMD を読み戻す結線テスト。

他の CLI 系テストは vpr の読み込みを差し替えて配線と終了コードを検証するため、CLI と実 read の
結線だけが検証から漏れる。ここは差し替えを一切挟まず、合成した最小の vpr(zip)を実ファイルとして
書き、そのパスで `cli.main` を実行して VMD が読み戻せるところまでを通す。vpr 解析そのものの網羅は
vpr のテストが担うので、ここでは経路が繋がっていることだけを見る。
"""

import io
import json
import zipfile

from vmd import read as vmd_read
from vpr2vmd import cli


def _write_vpr(path, notes):
    """sequence.json だけを持つ最小の vpr(zip)を path へ書く。"""
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
            "type": 2,  # 歌唱トラック(vpr が写像する種別)
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
    # 実ファイルの vpr を入力に、読み込みからモーフキーの書き出しまでが繋がっていること。
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
    # 正常な変換は標準エラーへ何も出さない。出力ポリシー(0F 中立登録→正規化)の副作用である
    # キーの並べ替えが警告として毎回漏れ出すと、本当の異常が埋没する。
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
    # 非 vpr の実バイト列を実 read へ通す経路。他の異常系テストは read を差し替えるため、
    # CLI と実 read を繋いだ失敗の分類はここでしか見られない。
    src = tmp_path / "in.vpr"
    src.write_bytes(b"this is not a zip archive")
    out = tmp_path / "out.vmd"

    assert cli.main([str(src), "-o", str(out)]) == 1

    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err
    assert not out.exists()
