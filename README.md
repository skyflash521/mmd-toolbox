# mmd-toolbox

MikuMikuDance (MMD) 向けのユーティリティツール群。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `docs/specs/` | MMDファイル形式の詳細仕様(VMD/PMX)。バイナリレイアウトの正 |
| `mmd_toolbox/` | 共通ライブラリ。VMD読み書き・補間曲線評価・カメラ座標変換 |
| `shakevmd/` | カメラVMDに手ぶれを焼き込むCLIツール |

仕様書は各ディレクトリ直下に置かれている:

- [mmd_toolbox/mmd_toolbox.md](mmd_toolbox/mmd_toolbox.md) — 共通ライブラリ全体
  - [mmd_toolbox/vmd-io.md](mmd_toolbox/vmd-io.md) — VMD読み書き・正規化
  - [mmd_toolbox/vmd-interp.md](mmd_toolbox/vmd-interp.md) — 補間曲線評価
  - [mmd_toolbox/vmd-camera.md](mmd_toolbox/vmd-camera.md) — カメラ座標変換
- [shakevmd/shakevmd.md](shakevmd/shakevmd.md) — shakevmd

## 開発環境

### コア(ビルド・テスト)

本体ライブラリと CLI のビルド・テストに必要な最小構成。実行時依存は `numpy` のみで OS 非依存。

| ツール | 用途 | 備考 |
|---|---|---|
| Python 3.11 以上 | 実装・テスト実行 | Windows は既定の `python` が 3.11 未満のことがあるため `py -3` を使う |
| Git | バージョン管理 | Windows は Git for Windows(Git Bash 同梱)を推奨 |

`numpy`(実行時依存)と `pytest`(開発依存)は `pip install -e ".[dev]"` で導入される。

### 環境構築

リポジトリルートで仮想環境を作成し、開発インストールする。

**macOS / Linux**

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

cmd の場合、有効化のみ `.\.venv\Scripts\activate.bat` に読み替える。

### AI 支援開発ワークフロー(任意)

本リポジトリの開発プロセス(TDD・Codex によるレビューループ・コミット委譲)は Claude Code と Codex を併用する(`.claude/` 配下のスキル・フック)。これらを使う場合は以下も導入する。

| ツール | インストール | 用途 |
|---|---|---|
| Node.js 20 以上(LTS 推奨) | 公式インストーラ / nvm | Codex companion(レビューループ)の実行 |
| Claude Code | `npm install -g @anthropic-ai/claude-code` | AI 開発ハーネス(スキル・フック・レビュー) |
| Codex CLI | `npm install -g @openai/codex` | コードレビューの実行体。**既定モデルに追従するため最新版を推奨**(古いとモデル非対応で API エラーになる) |
| Codex プラグイン | Claude Code 内で `/plugin marketplace add openai/codex-plugin-cc` の後 `/plugin install codex@openai-codex` | Claude Code から Codex を呼ぶ連携(`codex-review-loop` スキルが使用) |

**Windows 追加要件**: Claude Code は Bash ツール・フック・`watchdog.sh` の実行に **Git Bash** を使う(Git for Windows 同梱)。Git Bash が無いと Codex レビューループやフックが動作しないため、Windows では Git Bash の導入が必須。

## テスト実行

```sh
pytest            # 全テスト
pytest mmd_toolbox     # mmd_toolboxのみ
```

テストは外部サービス・ネットワーク・MMD本体を必要としない。

Windows でシンボリックリンク作成権限(開発者モードまたは管理者)が無い場合、
上書きガードの symlink テストは自動的に skip される(失敗にはならない)。

### MMD産テストデータについて

一部のテストはMMD本体で作成したVMDファイルを使用する。
未作成の場合、該当テストは
「MMD産テストデータがありません」というメッセージで失敗する。
作成手順(必要なファイルと条件)は
[mmd_toolbox/tests/data/README.md](mmd_toolbox/tests/data/README.md) を参照。
