# 検証手順

変更を確定させる前に通す検証の正本。検査の一覧と合格条件を定めるのはこの文書だけで、他の文書は
それらを再掲せずここを参照する。

## 1. 実行する検査

リポジトリルートで次の4つを実行する。すべてエラー0件で合格とする。

| 検査 | コマンド | 対象 |
|---|---|---|
| テスト | `pytest` | [pyproject.toml](../../pyproject.toml) の `testpaths` が指す全パッケージ・全ツール |
| 静的検査 | `ruff check` | リポジトリ内の Python 全体(gitignore 済みのパスは ruff が自動で外す) |
| リンク検査 | `lychee --config lychee.toml . .claude` | Markdown のリンク先の実在 |
| 節参照検査 | `python scripts/check_section_references.py` | 平文の節参照・裸のファイル参照が残っていないこと([cross-references.md](cross-references.md) の記法) |

`pytest` と `ruff check` はディレクトリを渡すと対象を絞れる。作業中の確認には絞ってよいが、合格判定は
絞らずに行う。

## 2. 導入

`pytest`・`ruff` は開発依存として入る。導入手順は [README.md](../../README.md) の開発環境を参照。
`lychee` は pip では入らないため各自で導入する。

## 3. CI との関係

[test.yml](../../.github/workflows/test.yml) が pull request でこの4つを走らせる(テストは Python 3.12 と
3.14 の両方)。ジョブはテスト・静的検査・リンク検査の3つで、節参照検査はリンク検査のジョブに同居する。
検査を増減させたときは、この文書とワークフローの両方を合わせる。
