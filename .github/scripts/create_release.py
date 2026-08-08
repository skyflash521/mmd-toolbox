"""タグ push から GitHub Release を作成する。

タグ名 <ツール>/v<バージョン> を受け取り、tools/<ツール>/CHANGELOG.md から該当バージョンの節を抽出して
Release の本文にする。形式逸脱・節なしの場合は失敗し、Release を作らない。
"""
import subprocess
import sys
from pathlib import Path

from changelog_format import FormatError, parse_sections


def main() -> None:
    tag = sys.argv[1]
    tool, _, version = tag.partition("/v")
    if not tool or not version:
        sys.exit(f"タグ名が <構成要素>/v<バージョン> の形式でない: {tag}")
    if not (Path("tools") / tool / "README.md").exists():
        # 公開CLIツール以外の構成要素のタグ。Release は公開ツールのみ作る。
        print(f"{tag} は公開CLIツールのタグでないため Release を作らない")
        return
    changelog = Path("tools") / tool / "CHANGELOG.md"
    if not changelog.exists():
        sys.exit(f"CHANGELOG が無い: {changelog}")
    try:
        sections = parse_sections(changelog)
    except FormatError as exc:
        sys.exit(f"{changelog}: {exc}")
    if version not in sections:
        sys.exit(f"{changelog} にバージョン {version} の節が無い")
    notes_file = Path("release_notes.md")
    notes_file.write_text(sections[version] + "\n", encoding="utf-8")
    subprocess.run(
        ["gh", "release", "create", tag,
         "--title", f"{tool} v{version}",
         "--notes-file", str(notes_file)],
        check=True,
    )


if __name__ == "__main__":
    main()
