# ツール公開手順

各CLIツールをリリースするときの標準手順。初回公開も、以降のバージョン更新リリースも本書に従う。版番号の付け方は [versioning.md](versioning.md) を正とし、本書はリリース時に踏む作業を定める。`<tool>` は対象ツールのパッケージ名に読み替える。

## 原則

- 公開対象以外のツール・共通ライブラリには触れない。
- 正本ドキュメント([versioning.md](versioning.md) / 対象ツールの仕様書)を先に整えてから、実装・テストと整合させる。
- リリース作業一式(A〜C)は develop ブランチ上で行い、develop から main への PR を経て確定する(D)。複数ツールの同時リリースは1つの PR で行ってよい。
- コミット手順はリポジトリの開発ルールに従う。各まとまりごとにコミットする。

## A. 版付け

版番号の付与・正本の所在・表示内容はすべて [versioning.md](versioning.md) に従う。本節はその「作業」を列挙するに留め、規約自体は再掲しない。

1. `<tool>/__init__.py` の `__version__` を更新する。初回は付与し(初期版の選び方は [versioning.md §4](versioning.md#4-初期版の付与))、以降のリリースでは該当桁を上げる(どの桁を上げるかは [versioning.md §3](versioning.md#3-どの桁をいつ上げるか))。
2. CLI に `--version` を備える。初回はこのオプションを追加する(表示内容と番号源は [versioning.md §1](versioning.md#1-管理単位))。表示は対象ツール自身の版のみとする(共通ライブラリは版を持たない。[versioning.md §1](versioning.md#1-管理単位))。
3. ドキュメントに版番号を記載している箇所(トップ [`README.md`](../../README.md) のツール表のバージョン列など)を、現在の `__version__` に合わせて更新する(番号源は [versioning.md §1](versioning.md#1-管理単位) のとおり `__version__` 一本)。
4. `pyproject.toml` の version をその日の CalVer へ更新する(値の付け方は [versioning.md §1](versioning.md#1-管理単位))。

## B. 公開前検証

1. **テスト緑**: `pytest <tool>` と全体を実行し緑を確認する。
2. **リンク検査**: `lychee --config lychee.toml . .claude` と `python .github/scripts/check_section_references.py` を実行しエラー0件を確認する。
3. **CLI スモークテスト**: 代表入力で end-to-end 実行する。終了コード0・出力の実体(サイズ・件数)・`--version`・`--help`・`--dry-run` を確認。前景で exit と生成物の実体を見届けてから合格判断する。生成物は利用者から見える所定ディレクトリに置く。
4. **配布物の確認**: `pyproject.toml` の `project.scripts` に対象コマンド、`packages.find` に対象パッケージが含まれることを確認。開発インストールで `<tool> --version` が通ること。

## C. 利用者向けドキュメント

1. `<tool>/README.md` を整える。初回は新規作成し、以降のリリースでは現状に合わせて更新する。利用者目線で次を含める:
   - 冒頭に「何ができるか」(手段でなく目的)。
   - 最小コマンド・主なオプション・入出力の説明。
   - 非目標と既知の制約。
   - 詳細は仕様の重複を避け、対象ツールの仕様書へ誘導。
2. `<tool>/CHANGELOG.md` を更新する(形式・粒度・書き方・初回リリース時の作成・README からのリンクは [changelog.md](changelog.md) を正とする)。
3. トップ [`README.md`](../../README.md) の対象ツールの記載を整える(ツールの列挙は名前順)。初回は次を追記し、以降のリリースでは既存の記載を更新する:
   - ツール表へ行(できること・`<tool>/README.md` リンク)。
   - 対象ツールの概要節(目的・最小コマンド・要点)。
   - リポジトリ構成表は `tools/<ツール>/` の汎用行が既に対象ツールを含むため、個別の行は追加しない。
   - 依存・開発環境に対象ツール固有の事項があれば反映(無ければ変更不要)。
4. 対象ツールの仕様書を公開水準に点検する: 目的・非目標・CLI仕様・既知の制約が現実装と整合するか。作業過程参照(ラウンド番号・計画書参照)が残っていれば除去([artifact-hygiene.md §2](artifact-hygiene.md#2-作業過程参照の混入禁止) が混入禁止対象の正本)。ツール仕様書は恒久仕様書なので、[document-authoring.md §2](document-authoring.md#2-恒久仕様書の記述範囲) の観点でロールアウト過程が残っていないかも点検。→ [cross-references.md](cross-references.md)・[terminology.md](terminology.md) の観点で参照の記法・用語規約を直接確認(リンク先の実在は B の手順2のリンク検査で確認する)。

## D. リリース確定

1. すべて緑・収束後、develop から main への PR を作成し、CI(PR で走る検査)が緑であることを確認する。
2. マージは明示指示で行う(自動でマージしない)。
3. main のマージコミットへ、リリースする CLIツールごとのタグを打ち、push する(タグの形式・種別・版の一致・同時リリース時の別タグは [versioning.md §5](versioning.md#5-タグ付け))。タグ付け・push は明示指示で行う(自動で打たない)。

   ```sh
   git push origin <tool>/v<MAJOR.MINOR.PATCH>
   ```

4. タグの push をトリガに、CI が GitHub Release を作成して CHANGELOG の該当版の節を本文へ転記する(転記の規則は [changelog.md §4](changelog.md#4-github-releases-への転記))。CI の workflow が成功し Release が作られたことを確認する。

## 順序の目安

A(版)→ B(検証)→ C(ドキュメント)を develop 上で一巡し、各まとまりでコミット。D(PR → マージ → タグ)は最後。
