"""PR 時に CHANGELOG を検査する。

常に全公開ツールの CHANGELOG の形式を検査する。さらに main 宛ての PR では、リリース対象
(main と比べて __version__ が変わった公開ツール。複数可)ごとに、その後のタグ push で
Release 本文を抽出できること(該当バージョンの節が先頭にある)を検査する。失敗は exit 1。
公開ツール = tools/<ツール>/ に利用者向け README.md を持つツール。
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from changelog_format import FormatError, parse_sections

_VERSION_RE = re.compile(r'__version__\s*=\s*"([^"]+)"')


def version_of(text: str | None) -> str | None:
    if text is None:
        return None
    found = _VERSION_RE.search(text)
    return found.group(1) if found else None


def version_on_main(tool: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"origin/main:tools/{tool}/__init__.py"],
        capture_output=True, text=True, encoding="utf-8",
    )
    return version_of(proc.stdout) if proc.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="", help="PR のベースブランチ名")
    args = parser.parse_args()

    failures: list[str] = []
    all_sections: dict[str, dict[str, str]] = {}
    for changelog in sorted(Path("tools").glob("*/CHANGELOG.md")):
        try:
            all_sections[changelog.parent.name] = parse_sections(changelog)
        except FormatError as exc:
            failures.append(f"{changelog}: {exc}")

    if args.base == "main":
        for readme in sorted(Path("tools").glob("*/README.md")):
            tool = readme.parent.name
            version = version_of((readme.parent / "__init__.py").read_text(encoding="utf-8"))
            if version is None:
                failures.append(f"tools/{tool}/__init__.py から __version__ を読めない")
                continue
            if version == version_on_main(tool):
                continue  # バージョンが変わっていない = このPRのリリース対象でない
            changelog = Path("tools") / tool / "CHANGELOG.md"
            sections = all_sections.get(tool)
            if sections is None:
                failures.append(f"リリース対象 {tool} v{version}: {changelog} が無いか形式不正")
            elif version not in sections:
                failures.append(f"リリース対象 {tool} v{version}: {changelog} にバージョン {version} の節が無い")
            elif next(iter(sections)) != version:
                failures.append(f"リリース対象 {tool} v{version}: {changelog} の先頭の節が {version} でない")
            else:
                print(f"リリース対象 {tool} v{version}: CHANGELOG 検査 OK")

    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print(f"CHANGELOG 検査 OK({len(all_sections)}件)")


if __name__ == "__main__":
    main()
