"""song2vpr CLI 骨組みのテスト。

範囲は CLI の起動・引数解析・検証・出力先解決・上書きガードと、処理を伴わない経路に限る。
音声前段の呼び出し以降は後続の配線で入るため、ここでは扱わない。

終了コード: 0 正常 / 1 入力不正 / 2 引数エラー(未知オプション・範囲不正・上書きガード等) /
3 出力書き込み失敗 / 4 音声前段の外部依存の失敗・追加依存の未導入 / 130 協調的な中断。
"""

import re

import pytest

cli = pytest.importorskip("song2vpr.cli", reason="CLI モジュールがまだ無い")


def _touch(path):
    path.write_bytes(b"")
    return str(path)


# --- 主要オプションの受理 ----------------------------------------------------


def test_parses_full_option_set(tmp_path):
    """統一フラグと共通引数群を一通り受理する。"""
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vpr"
    assert cli.main([
        src,
        "-o", str(out),
        "--separate-vocals", "always",
        "--separator", "audio-separator-htdemucs-ft",
        "--recognizer-model-id", "openai/whisper-medium",
        "--recognizer-model-revision", "abc123",
        "--recognizer-retry",
        "--forced-aligner", "wav2vec2-ctc-forcedalign",
        "--english-katakana-method", "arpakana",
        "--device", "cpu",
        "--max-duration", "120",
        "--keep-intermediate",
        "--quiet",
        "-v",
        "--dry-run",
    ]) == 0


def test_dry_run_writes_no_output(tmp_path):
    """--dry-run は出力 vpr を書かない(書き出し経路が入るまでは通常実行との差が出ない)。"""
    src = _touch(tmp_path / "in.wav")
    out = tmp_path / "out.vpr"
    assert cli.main([src, "-o", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_default_output_is_vpr_alongside_input(tmp_path):
    """既定出力は <入力名>.vpr。同名の既存ファイルが上書きガードに掛かることで解決先を確かめる。"""
    src = _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vpr")
    assert cli.main([src, "--dry-run"]) == 2


def test_default_output_does_not_collide_with_other_extension(tmp_path):
    """既定出力は .vpr なので、別拡張子の同名ファイルは上書きガードに掛からない。"""
    src = _touch(tmp_path / "song.wav")
    _touch(tmp_path / "song.vmd")
    assert cli.main([src, "--dry-run"]) == 0


# --- 引数エラー(終了コード2)-------------------------------------------------


def test_unknown_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--bogus"]) == 2


def test_missing_positional_is_arg_error():
    assert cli.main([]) == 2


def test_unknown_separate_vocals_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separate-vocals", "sometimes", "--dry-run"]) == 2


def test_unknown_separator_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separator", "nope", "--dry-run"]) == 2


def test_unknown_forced_aligner_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--forced-aligner", "unknown-aligner", "--dry-run"]) == 2


def test_unknown_device_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--device", "gpu", "--dry-run"]) == 2


def test_recognizer_model_revision_without_model_id_is_arg_error(tmp_path):
    """--recognizer-model-revision だけの指定は対象が無く無意味なので引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-revision", "abc123", "--dry-run"]) == 2


@pytest.mark.parametrize("value", ["always", "never"])
def test_separate_vocals_choices_are_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--separate-vocals", value, "--dry-run"]) == 0


@pytest.mark.parametrize("value", ["auto", "cpu"])
def test_device_choices_are_accepted(tmp_path, value):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--device", value, "--dry-run"]) == 0


def test_unknown_english_katakana_method_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--english-katakana-method", "nope", "--dry-run"]) == 2


def test_no_recognizer_retry_is_accepted(tmp_path):
    """既定 on の否定形 --no-recognizer-retry も受理する(--x/--no-x の対)。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--no-recognizer-retry", "--dry-run"]) == 0


def test_sofa_timeout_positive_value_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--sofa-timeout", "60", "--dry-run",
    ]) == 0


def test_recognizer_model_id_accepts_any_string(tmp_path):
    """--recognizer-model-id は安定 id でなく自由な文字列。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--recognizer-model-id", "anything/goes", "--dry-run"]) == 0


@pytest.mark.parametrize("missing", ["--sofa-python", "--sofa-root", "--sofa-checkpoint"])
def test_forced_aligner_sofa_missing_required_arg_is_arg_error(tmp_path, missing):
    """SOFA 経路は必須3項目が揃わないと引数エラー(組み合わせ検証は共有側が持つ)。"""
    src = _touch(tmp_path / "in.wav")
    given = {"--sofa-python": "/venv/python", "--sofa-root": "/sofa", "--sofa-checkpoint": "/ckpt.ckpt"}
    del given[missing]
    argv = [src, "--forced-aligner", "sofa-forcedalign", "--dry-run"]
    for name, value in given.items():
        argv += [name, value]
    assert cli.main(argv) == 2


def test_forced_aligner_sofa_with_all_required_args_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([
        src, "--forced-aligner", "sofa-forcedalign",
        "--sofa-python", "/venv/python", "--sofa-root", "/sofa",
        "--sofa-checkpoint", "/ckpt.ckpt", "--dry-run",
    ]) == 0


def test_max_duration_zero_is_accepted(tmp_path):
    """--max-duration 0 は分割しない指定として有効値。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--max-duration", "0", "--dry-run"]) == 0


def test_max_duration_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "--max-duration=-1", "--dry-run"]) == 2


@pytest.mark.parametrize("opt, value", [
    ("--max-duration", "abc"), ("--max-duration", "nan"), ("--sofa-timeout", "abc"),
    ("--sofa-timeout", "0"),
])
def test_option_value_error_states_the_reason_in_japanese(tmp_path, opt, value, capsys):
    # 引数エラーの理由は利用者に伝わる日本語で出す。検証子の実体がそのまま文字列化されると、
    # 実行ごとに変わるアドレスが理由の代わりに出て何も伝わらない。
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, opt, value, "--dry-run"]) == 2
    line = capsys.readouterr().err
    assert re.search(r"[぀-ヿ一-鿿]", line)
    assert "0x" not in line and " object at " not in line


# --- 上書きガード・出力先の検査 ----------------------------------------------


def test_existing_output_requires_overwrite(tmp_path):
    """出力先に既存ファイルがあると --overwrite 無しは引数エラー。"""
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out]) == 2


def test_existing_output_allowed_with_overwrite(tmp_path):
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out, "--overwrite", "--dry-run"]) == 0


def test_overwrite_guard_is_evaluated_in_dry_run(tmp_path):
    """空実行でも上書きガードは評価する(出力を書かないことと検証を省くことは別)。"""
    src = _touch(tmp_path / "in.wav")
    out = _touch(tmp_path / "out.vpr")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2


def test_missing_output_path_is_not_blocked(tmp_path):
    """出力先が存在しなければ --overwrite 無しで通る。"""
    src = _touch(tmp_path / "in.wav")
    assert cli.main([src, "-o", str(tmp_path / "out.vpr"), "--dry-run"]) == 0


@pytest.mark.parametrize("extra", [[], ["--overwrite"]])
def test_output_directory_is_rejected_regardless_of_overwrite(tmp_path, extra):
    """出力先が既存ディレクトリなら --overwrite の有無に依らず専用コードで拒否する。"""
    src = _touch(tmp_path / "in.wav")
    outdir = tmp_path / "dir"
    outdir.mkdir()
    assert cli.main([src, "-o", str(outdir), *extra, "--dry-run"]) == 2


# --- ヘルプ ------------------------------------------------------------------


def test_help_lists_key_flags(capsys):
    """--help だけで使い方を把握できるよう、主要な引数を列挙する。"""
    assert cli.main(["--help"]) == 0
    text = capsys.readouterr().out
    for flag in ("--output", "--overwrite", "--dry-run", "--keep-intermediate", "--verbose",
                 "--quiet", "--machine", "--describe", "--version",
                 "--separate-vocals", "--separator", "--recognizer-model-id",
                 "--recognizer-model-revision", "--recognizer-retry", "--forced-aligner",
                 "--sofa-python", "--sofa-root", "--sofa-checkpoint", "--sofa-timeout",
                 "--english-katakana-method", "--device", "--max-duration"):
        assert flag in text
