"""失敗報告の共通経路。

構造化出力モードの error イベントと人間向けのエラー行を 1 つの入口から出し、標準出力へ書けない
場合の後退まで含めて各 CLI が同じ挙動になるようにする。
"""

import sys

from .events import error_event

# 書き込み先が使えないことを示す例外。呼び出し側がパイプを先に閉じた場合は OSError、標準出力
# 自体が閉じられている場合は ValueError になる。これ以外の例外はイベントの中身の不備など
# 書き込み先の問題ではない失敗なので、後退させず呼び出し側へ渡す(終端イベントを出せる状況で
# 黙ってストリームを途切れさせないため)。
_UNUSABLE_STREAM = (OSError, ValueError)


def emit_failure(emitter, *, code, message, exit_code, field=None, path=None, stderr=None,
                 **extra):
    """失敗を 1 回だけ報告し、受け取った exit_code をそのまま返す。

    emitter が None(構造化出力モードでない・エミッタ未確立)か既に終端済みなら、人間向けの
    エラー行を 1 行出す。終端済みの場合、既に出した終端イベントは書き換えない(result 送出の
    直後に中断・想定外例外が起きた場合の二次的な失敗を握り潰さないため、返り値は今回受け取った
    exit_code のままにする)。それ以外は error イベントを送出して終端し、書き込み先が使えず
    失敗した場合は同じエラー行へ切り替える。

    extra は error イベントへ載せる追加キー(ツール固有のペイロード)。人間向けのエラー行には
    載せない。stderr はエラー行の出力先で、既定の None は呼び出しの時点で標準エラーへ解決する
    (定義時に束縛すると実行時の差し替え・捕捉が効かない)。
    """
    if emitter is not None and not emitter.terminated:
        event = error_event(code=code, message=message, exit_code=exit_code, field=field, path=path)
        event.update(extra)
        try:
            emitter.error(**event)
        except _UNUSABLE_STREAM:
            _write_error_line(message, stderr)
        return exit_code
    _write_error_line(message, stderr)
    return exit_code


def _write_error_line(message, stderr):
    """人間向けのエラー行を 1 行出す。書き込みの失敗は握り潰す。

    標準エラーへ書けないことで終了コード・出力を変えない(人間向け標準エラーの書き込み失敗で
    プロセスを落とさない)。
    """
    try:
        print(f"error: {message}", file=sys.stderr if stderr is None else stderr)
    except Exception:
        pass
