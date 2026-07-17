#!/usr/bin/env python3
"""経過時間の上限を超えたら子プロセスを強制終了する実行ラッパー。

外側ランナー(呼び出し元シェルのtimeout機構)や自動背景化に頼ると、背景化した子プロセスが
オーファン化してkillされず上限を超えて走り続けることがある。このラッパーはコマンド自身の
中で上限を持ち、子プロセスを止める(Windowsはプロセスツリー全体を taskkill /F /T、それ以外は
開始したプロセスグループへ SIGKILL。子孫プロセスが独自のセッション/プロセスグループへ移った
場合はその限りではない)。**キル操作自体が失敗しても、このラッパー(と呼び出し側)は無期限には
待たない**: キル後は短い上限で1回だけ追加で待ち、それでも終了しなければ失敗として報告して
復帰する。子の標準出力・標準エラーはこのラッパーへパイプで受けて自身の標準出力・標準エラーへ
逐次中継する(子にラッパー自身のfdを直接継承させない)。継承させると、killに失敗して子が
生き残った場合に子がfdを保持し続け、ラッパー自身が復帰してもパイプの書き込み側が閉じきらず、
呼び出し側(このラッパーの出力を読む側)が子の終了まで待たされてしまうため。

Usage: python3 run_capped.py <cap_seconds> -- <command...>
Exit code: 124 if the cap was exceeded (regardless of whether the kill itself
fully succeeded), otherwise the child's exit code.
"""
import math
import os
import subprocess
import sys
import threading

_POST_KILL_WAIT_SEC = 10.0
_RELAY_JOIN_TIMEOUT_SEC = 5.0


def _resolve_executable(token):
    """パス区切りを含む相対パスは絶対パスへ解決して返す。Windowsのsubprocess.Popen
    (shell=False)は相対パス(特にスラッシュ区切り)の実行ファイルを見つけられないことが
    あるため。パス区切りを含まないbare名(PATH検索に委ねるコマンド名)はそのまま返す。"""
    if ("/" in token or "\\" in token) and not os.path.isabs(token):
        return os.path.abspath(token)
    return token


def _parse_args(args):
    if len(args) < 2 or args[1] != "--":
        return None, None
    try:
        cap_seconds = float(args[0])
    except ValueError:
        return None, None
    if not math.isfinite(cap_seconds) or cap_seconds <= 0:
        return None, None
    command = args[2:]
    if not command:
        return None, None
    command = [_resolve_executable(command[0])] + command[1:]
    return cap_seconds, command


def _relay(src, dst):
    """read(n) はバッファ層がnバイト分溜まるかEOFまで戻らないことがあり、子プロセスが
    少量ずつ出力しては待つ場合に中継が遅延する。read1(n) は下層ストリームへの読み出しを
    高々1回しか行わないため、その時点で利用可能な分だけをすぐ返す(真に逐次的な中継)。
    """
    try:
        for chunk in iter(lambda: src.read1(4096), b""):
            dst.write(chunk)
            dst.flush()
    except (OSError, ValueError):
        pass
    finally:
        try:
            src.close()
        except OSError:
            pass


def _wait_after_kill(proc):
    try:
        proc.wait(timeout=_POST_KILL_WAIT_SEC)
    except subprocess.TimeoutExpired:
        print("[run_capped] process still alive after kill attempt; giving up without "
              "blocking indefinitely", file=sys.stderr)
    except OSError as exc:
        print(f"[run_capped] failed to wait for process: {exc}", file=sys.stderr)


def _kill_windows(proc):
    try:
        result = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                 capture_output=True, text=True, timeout=_POST_KILL_WAIT_SEC)
        if result.returncode != 0:
            print(f"[run_capped] taskkill failed (code {result.returncode}): "
                  f"{result.stderr.strip()}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("[run_capped] taskkill itself timed out", file=sys.stderr)
    except OSError as exc:
        print(f"[run_capped] failed to invoke taskkill: {exc}", file=sys.stderr)
    _wait_after_kill(proc)


def _kill_posix(proc):
    import os
    import signal

    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # 既に終了している(TimeoutExpired直後の競合)
    except OSError as exc:
        print(f"[run_capped] failed to send SIGKILL: {exc}", file=sys.stderr)
    _wait_after_kill(proc)


def main():
    args = sys.argv[1:]
    cap_seconds, command = _parse_args(args)
    if cap_seconds is None:
        print("usage: run_capped.py <cap_seconds> -- <command...>", file=sys.stderr)
        return 2

    popen_kwargs = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             **popen_kwargs)
    stdout_thread = threading.Thread(target=_relay, args=(proc.stdout, sys.stdout.buffer),
                                      daemon=True)
    stderr_thread = threading.Thread(target=_relay, args=(proc.stderr, sys.stderr.buffer),
                                      daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = proc.wait(timeout=cap_seconds)
        stdout_thread.join(timeout=_RELAY_JOIN_TIMEOUT_SEC)
        stderr_thread.join(timeout=_RELAY_JOIN_TIMEOUT_SEC)
        return returncode
    except subprocess.TimeoutExpired:
        print(f"[run_capped] exceeded cap ({cap_seconds:.0f}s); killing", file=sys.stderr)
        if sys.platform == "win32":
            _kill_windows(proc)
        else:
            _kill_posix(proc)
        # 中継スレッドはデーモンなので join を待たずにプロセスごと終了してよい
        # (子が生き残っていてもラッパー自身のfdはここで閉じ、呼び出し側は復帰できる)。
        return 124


if __name__ == "__main__":
    sys.exit(main())
