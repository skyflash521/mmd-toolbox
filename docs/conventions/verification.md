# 検証手順

変更を確定させる前に通す検査の一覧と合格条件の正本。

## 1. 実行する検査

リポジトリルートで次の5つを実行する。すべてエラー0件で合格とする。

| 検査 | コマンド | 対象 |
|---|---|---|
| 静的検査 | `ruff check` | リポジトリ内の Python 全体(gitignore 済みのパスは ruff が自動で外す) |
| 記法検査 | `rumdl check` | Markdown の一般的な記法(見出し・リスト・コードブロックまわりの体裁)。gitignore 済みのパスは rumdl が自動で外す |
| 節参照検査 | `python scripts/check_section_references.py` | 平文の節参照・裸のファイル参照が残っていないこと |
| リンク検査 | `lychee --config lychee.toml . .claude` | Markdown のリンク先の実在 |
| テスト | `pytest` | [pyproject.toml](../../pyproject.toml) の `testpaths` が指す全パッケージ・全ツール |

`pytest`・`ruff check`・`rumdl check` はディレクトリを渡すと対象を絞れる。作業中の確認には絞って
よいが、合格判定は絞らずに行う。

## 2. 実素材を使う人手検査

実際の歌唱素材(楽曲音源と、人が作った正解 vpr)を使う検査。素材はリポジトリに含まれない。CI では
走らせず、実素材を扱う変更を確定させる前に人が実行する。

| 検査 | コマンド | 合格条件 |
|---|---|---|
| 出力 vpr の機械検査 | `python scripts/check_vpr_output.py <生成した vpr> <正解 vpr> [<正解 vpr> ...]` | 違反 0 件(何も出力せず終了コード 0) |

## 3. 導入

`pytest`・`ruff`・`rumdl` は開発依存として入る。`lychee` は pip では入らず、各自で導入する。

## 4. CI との関係

[test.yml](../../.github/workflows/test.yml) が pull request で走らせる検査は [§1](#1-実行する検査) と一致させる。
