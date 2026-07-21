"""外部ツール(分離器・認識モデル)の進捗表示・ログを一括で抑える。

呼び出し側 CLI の進捗表示(改行なしで同じ行を上書きするライブ行)と同じ標準エラーへ、これらの
外部ライブラリ(Demucs・transformers・huggingface_hub・tqdm)が抑制されないままログ・進捗バーを
書き込むと、行が連結して読めなくなる。分離・認識の前に一度呼べば足り、複数回呼んでも安全
(2回目以降は何もしない)。pyopenjtalk 由来の警告(ネイティブ拡張が標準エラーへ直接書き込むため、
Python 側のログ設定では抑えられない)は suppress_native_stderr で個別に囲む。
"""

import contextlib
import os
import sys
from functools import partialmethod

_WRITE_ERRORS = (OSError, ValueError, UnicodeError)

_done = False


@contextlib.contextmanager
def suppress_native_stderr():
    """OS レベルで標準エラー(fd 2)を一時的に抑える。

    pyopenjtalk(open_jtalk のネイティブ拡張)は njd2jpcommon.c から Python のログ設定を
    経由せず fd へ直接警告を書き込むため、Python 側の logging/warnings では抑えられない。
    この間は他スレッドが同じ fd へ書く出力(呼び出し側 CLI の進捗表示等)も一時的に抑えられるが、
    対象呼び出し(1回の g2p 変換)は短時間で終わるため実害は小さい。fd の復元(dup2・close)は
    flush の成否に関わらず必ず先に行う(flush が失敗すると fd が devnull を指したまま残り、
    以後の標準エラー出力が全て失われるため)。
    """
    try:
        sys.stderr.flush()
    except _WRITE_ERRORS:
        pass
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)
        try:
            sys.stderr.flush()
        except _WRITE_ERRORS:
            pass


def silence_third_party_output():
    global _done
    if _done:
        return
    _done = True

    try:
        from tqdm import tqdm

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    except ImportError:
        pass

    try:
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import logging as hf_logging

        disable_progress_bars()
        hf_logging.set_verbosity(hf_logging.CRITICAL)
    except ImportError:
        pass

    try:
        import transformers

        transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)
    except ImportError:
        pass
