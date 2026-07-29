#!/usr/bin/env python3
"""Best-effort attention sound for Windows, macOS, and Linux.

Used when Claude needs the user: a permission prompt appeared, or the turn ended and
the conversation is waiting for input.

Sound playback is deliberately non-critical: missing commands, missing sound files,
unavailable audio devices, and all other failures are silently ignored. Players are
launched fire-and-forget so the caller is never delayed by playback.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

DEVNULL = subprocess.DEVNULL


def _spawn(command: list[str]) -> bool:
    """Launch one player fire-and-forget, returning whether it was spawned.

    The notification must not delay the permission prompt, so we never wait for
    the player to finish. Short notification sounds exit on their own.
    """
    try:
        subprocess.Popen(
            command,
            stdin=DEVNULL,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        return True
    except Exception:
        return False


def _play_windows() -> bool:
    try:
        import winsound

        for frequency in (880, 1175, 1568):
            winsound.Beep(frequency, 130)
        return True
    except Exception:
        return False


def _play_macos() -> bool:
    player = shutil.which("afplay")
    sound = Path("/System/Library/Sounds/Glass.aiff")
    return bool(player and sound.is_file() and _spawn([player, str(sound)]))


def _first_existing(paths: tuple[str, ...]) -> str | None:
    return next((path for path in paths if Path(path).is_file()), None)


def _play_linux() -> bool:
    canberra = shutil.which("canberra-gtk-play")
    if canberra:
        return _spawn([canberra, "--id=dialog-question", "--description=Claude needs attention"])

    desktop_sound = _first_existing(
        (
            "/usr/share/sounds/freedesktop/stereo/dialog-question.oga",
            "/usr/share/sounds/freedesktop/stereo/bell.oga",
            "/usr/share/sounds/alsa/Front_Center.wav",
        )
    )
    for player_name in ("paplay", "pw-play"):
        player = shutil.which(player_name)
        if player and desktop_sound:
            return _spawn([player, desktop_sound])

    wav_sound = _first_existing(
        (
            "/usr/share/sounds/alsa/Front_Center.wav",
            "/usr/share/sounds/alsa/Front_Left.wav",
        )
    )
    player = shutil.which("aplay")
    if player and wav_sound:
        return _spawn([player, wav_sound])
    return False


def _terminal_bell() -> None:
    try:
        for stream in (sys.stderr, sys.stdout):
            if stream and stream.isatty():
                stream.write("\a")
                stream.flush()
                return
    except Exception:
        pass


def main() -> None:
    system = platform.system()
    played = False

    if system == "Windows":
        played = _play_windows()
    elif system == "Darwin":
        played = _play_macos()
    elif system == "Linux":
        played = _play_linux()

    if not played:
        _terminal_bell()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    os._exit(0)
