# 検証手順

変更を確定させる前に通す検証の正本。検査の一覧と合格条件を定めるのはこの文書だけで、他の文書は
それらを再掲せずここを参照する。

## 1. 実行する検査

リポジトリルートで次の5つを実行する。すべてエラー0件で合格とする。

| 検査 | コマンド | 対象 |
|---|---|---|
| 静的検査 | `ruff check` | リポジトリ内の Python 全体(gitignore 済みのパスは ruff が自動で外す) |
| 記法検査 | `rumdl check` | Markdown の一般的な記法(見出し・リスト・コードブロックまわりの体裁)。gitignore 済みのパスは rumdl が自動で外す |
| 節参照検査 | `python scripts/check_section_references.py` | 平文の節参照・裸のファイル参照が残っていないこと([cross-references.md](cross-references.md) の記法) |
| リンク検査 | `lychee --config lychee.toml . .claude` | Markdown のリンク先の実在 |
| テスト | `pytest` | [pyproject.toml](../../pyproject.toml) の `testpaths` が指す全パッケージ・全ツール |

`pytest`・`ruff check`・`rumdl check` はディレクトリを渡すと対象を絞れる。作業中の確認には絞って
よいが、合格判定は絞らずに行う。

## 2. 導入

`pytest`・`ruff`・`rumdl` は開発依存として入る。導入手順は
[README.md の開発環境](../../README.md#開発環境) を参照。`lychee` は pip では入らないため各自で
導入する。

`rumdl` が外す規則とその理由は [pyproject.toml](../../pyproject.toml) の `[tool.rumdl]` に置く。

## 3. CI との関係

[test.yml](../../.github/workflows/test.yml) が pull request でこの5つを走らせる(テストは Python 3.12 と
3.14 の両方)。ジョブはテスト・静的検査・記法検査・リンク検査の4つで、節参照検査はリンク検査のジョブに
同居する。検査を増減させたときは、この文書とワークフローの両方を合わせる。
