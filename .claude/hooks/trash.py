#!/usr/bin/env python3
"""rm の安全な代替: 指定パスを削除せずOS標準のごみ箱へ送る(可逆)。Windows は
SHFileOperationW(FOF_ALLOWUNDO)、macOS は Finder への delete AppleScript、Linux は
XDG Trash 仕様(freedesktop.org)に従う。guard-rm.py が rm を deny し、これへ誘導する。

Usage: python3 .claude/hooks/trash.py <path>...
"""
import ctypes
import os
import pathlib
import subprocess
import sys
import urllib.parse
from datetime import datetime


def _trash_windows(paths):
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE = 0x0003
    FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_SILENT, FOF_NOERRORUI = 0x0040, 0x0010, 0x0004, 0x0400
    op = SHFILEOPSTRUCTW()
    op.pFrom = "\0".join(str(pathlib.Path(os.path.abspath(p))) for p in paths) + "\0\0"
    op.wFunc = FO_DELETE
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if result != 0 or op.fAnyOperationsAborted:
        raise RuntimeError(f"SHFileOperationW failed: code={result}")


def _applescript_quote(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _trash_macos(paths):
    items = ", ".join(
        f'POSIX file "{_applescript_quote(str(pathlib.Path(os.path.abspath(p))))}"' for p in paths
    )
    subprocess.run(
        ["osascript", "-e", f'tell application "Finder" to delete {{{items}}}'], check=True
    )


def _trash_linux(paths):
    home = pathlib.Path(os.environ.get("XDG_DATA_HOME", pathlib.Path.home() / ".local/share"))
    files_dir, info_dir = home / "Trash/files", home / "Trash/info"
    files_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    for raw in paths:
        src = pathlib.Path(os.path.abspath(raw))
        dest, n = files_dir / src.name, 1
        while os.path.lexists(dest):
            dest = files_dir / f"{src.stem}.{n}{src.suffix}"
            n += 1
        (info_dir / f"{dest.name}.trashinfo").write_text(
            f"[Trash Info]\nPath={urllib.parse.quote(str(src))}\n"
            f"DeletionDate={datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )
        src.rename(dest)


def main():
    missing = [p for p in sys.argv[1:] if not os.path.lexists(p)]
    for p in missing:
        print(f"skip (not found): {p}", file=sys.stderr)
    paths = [p for p in sys.argv[1:] if p not in missing]
    if not paths:
        print("no existing paths given", file=sys.stderr)
        return 1
    if sys.platform == "win32":
        _trash_windows(paths)
    elif sys.platform == "darwin":
        _trash_macos(paths)
    else:
        _trash_linux(paths)
    for p in paths:
        print(f"sent to trash: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
