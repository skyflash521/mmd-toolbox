"""song2vpr CLI 移植性のテスト。

非ASCIIパスの受理・既定出力の導出、相対パスの解決(暗黙のベースディレクトリを持たない)、機械モード
標準出力が非ASCIIを UTF-8 のまま出すこと、人間向け標準エラーの符号化安全性(ロケール符号化で表せない
文字でもプロセスを落とさない)を検証する。

範囲はこの骨組みで通る経路(引数解析・出力先解決・ガード)に限る。書き出しを伴う往復は音声前段の
配線後に検証する。
"""

import io
import json
import sys

import pytest

cli = pytest.importorskip("song2vpr.cli", reason="impl pending: T-1 パッケージ雛形とCLI骨組み")

# cp932(Windows のロケール符号化)で表せない文字(絵文字 U+1F3A5)。ロケール符号化外の文字を
# 人間向け標準エラーへ書く経路を作り、符号化安全性を検証するために使う。
UNREP = "\U0001f3a5"


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _cp932_stderr(monkeypatch):
    """sys.stderr をロケール符号化(cp932)相当・strict へ差し替える(符号化安全性の検証用)。"""
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


# --- 非ASCIIパスの受理と既定出力の導出 --------------------------------------


def test_non_ascii_input_resolves_default_output(tmp_path):
    """日本語名の入力でも既定出力は <入力名>.vpr(同名の既存ファイルが上書きガードに掛かる)。"""
    src = _touch(tmp_path / "ボーカル.wav")
    _touch(tmp_path / "ボーカル.vpr")
    assert cli.main([src, "--dry-run"]) == 2


def test_non_ascii_input_does_not_collide_with_other_extension(tmp_path):
    """既定出力は .vpr なので、日本語名でも別拡張子の同名ファイルは上書きガードに掛からない。"""
    src = _touch(tmp_path / "ボーカル.wav")
    _touch(tmp_path / "ボーカル.vmd")
    assert cli.main([src, "--dry-run"]) == 0


def test_non_ascii_output_path_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", str(tmp_path / "歌声パート.vpr"), "--dry-run"]) == 0


# --- 機械モード標準出力の符号化 ----------------------------------------------


def test_machine_stdout_keeps_non_ascii_as_utf8(tmp_path, capsysbinary):
    """イベントの非ASCII文字は UTF-8 のまま出す(\\uXXXX へ逃がさない)。"""
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "出力先ディレクトリ"
    outdir.mkdir()
    assert cli.main(["--machine", src, "-o", str(outdir), "--dry-run"]) == 2
    raw = capsysbinary.readouterr().out
    # ASCII へ逃がす実装だと日本語は退避表記になり、UTF-8 バイト列は現れない。
    assert "出力先ディレクトリ".encode() in raw
    events = [json.loads(ln) for ln in raw.decode("utf-8").split("\n") if ln]
    assert events[-1]["path"] == str(outdir)


# --- 人間向け標準エラーの符号化安全性 ----------------------------------------


def test_stderr_unencodable_char_does_not_crash(tmp_path, monkeypatch):
    """ロケール符号化で表せない文字を含む理由でも、符号化失敗で落とさず置換して出す。"""
    src = _touch(tmp_path / "in.wav")
    stderr = _cp932_stderr(monkeypatch)
    assert cli.main([src, "--separator", UNREP, "--dry-run"]) == 2
    stderr.flush()
    written = stderr.buffer.getvalue().decode("cp932")
    assert written  # 理由行が書けている
    # 表せない文字は落とさず退避表記へ置き換えて出す(符号化経路を実際に通ったことの確認)。
    assert "1f3a5" in written.lower()


# --- 相対パスの解決 ----------------------------------------------------------


def test_relative_input_is_resolved_against_cwd(tmp_path, monkeypatch):
    """相対パスは起動時の作業ディレクトリを基準に解決する(暗黙のベースディレクトリを持たない)。"""
    _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vpr")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["song.wav", "--dry-run"]) == 2


def test_default_output_follows_the_input_directory(tmp_path, monkeypatch):
    """既定出力は入力に隣接する(作業ディレクトリ直下ではない)。"""
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "song.wav")
    monkeypatch.chdir(tmp_path)
    # 入力に隣接する既定出力が既にあれば上書きガードに掛かる。
    _touch(sub / "song.vpr")
    assert cli.main(["sub/song.wav", "--dry-run"]) == 2


def test_default_output_ignores_same_name_in_cwd(tmp_path, monkeypatch):
    """作業ディレクトリ直下の同名ファイルは既定出力ではないのでガードに掛からない。"""
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "song.wav")
    _touch(tmp_path / "song.vpr")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["sub/song.wav", "--dry-run"]) == 0


def test_relative_output_is_resolved_against_cwd(tmp_path, monkeypatch):
    _touch(tmp_path / "in.wav")
    _touch(tmp_path / "out.vpr")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["in.wav", "-o", "out.vpr", "--dry-run"]) == 2
    assert cli.main(["in.wav", "-o", "other.vpr", "--dry-run"]) == 0
