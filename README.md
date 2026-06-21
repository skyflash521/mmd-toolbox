# mmd-toolbox

MikuMikuDance (MMD) 向けのユーティリティツール群。

## ツール

| ツール | できること | まず試すコマンド | 詳しい使い方 |
|---|---|---|---|
| `shakevmd` | カメラモーション VMD ファイルに手ぶれ効果を追加する | `shakevmd input.vmd` | [shakevmd/README.md](shakevmd/README.md) |

## 使い方

### 1. Python 3.11 以上をインストールする

- [Python 公式ダウンロードページ](https://www.python.org/downloads/)
- [Python 公式ドキュメント: Python のセットアップと利用](https://docs.python.org/ja/3/using/index.html)

Python のインストール手順は AI に次のように聞いてね。

```text
WindowsまたはmacOSでPython 3.11以上をインストールして、ターミナルでpythonコマンドが使えるようにする手順を、初心者向けに教えて。
```

### 2. mmd-toolbox をダウンロードする

1. このリポジトリのページで、緑色の `Code` ボタンを押す。
2. `Download ZIP` を押して ZIP ファイルを保存する。
3. Windows では保存した ZIP ファイルを右クリックして `すべて展開` を選ぶ。macOS では保存した ZIP ファイルをダブルクリックする。
4. 展開してできたフォルダを、次の手順で使う。

Git が使える人は clone して

```sh
git clone https://github.com/skyflash521/mmd-toolbox.git
```

### 3. README.md が入っているフォルダへ移動する

次は、Windows では PowerShell、macOS ではターミナルで、この README が入っているフォルダへ移動する。

1. PowerShell またはターミナルに `cd ` と入力する。`cd` の後ろには半角スペースを入れる。
2. 展開したフォルダを開き、`README.md` が見えるフォルダを PowerShell またはターミナルへドラッグ＆ドロップする。
3. Enter キーを押す。

Enter キーを押す前は、たとえば次のようになる。

```powershell
cd "C:\Users\ユーザー名\Downloads\mmd-toolbox-main\mmd-toolbox-main"
```

以降のコマンドは、このフォルダの中で実行する。

### 4. 初回準備コマンドを実行する

最後に、初回準備をする。
Windows PowerShell では、次のコマンドを上から順番に実行する。

```powershell
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install .
```

macOS では、次のコマンドを上から順番に実行する。

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## shakevmd: カメラに手ぶれ効果を追加する

`shakevmd` は、MMD のカメラモーション VMD ファイルに手ぶれ効果を追加するツール。
VMD は、MMD でモーションやカメラの動きを保存するファイル形式。ここで扱うのはカメラの動きが入った VMD。

使い方:

1. カメラモーション VMD ファイルを用意する。
   - MMD で `編集` → `カメラフレーム全て選択` を選んでから、`ファイル` → `モーションデータ保存` で VMD を保存する。
2. 次のコマンドを実行する。

```sh
shakevmd Camera.vmd
```

揺れを足した新しい VMD が `Camera_shake.vmd` という名前で作られる。

処理の特徴:

- 元のカメラの動きを1フレームごとに読み取り、帯域制限ノイズから作った回転・位置オフセットを加えてカメラキーとして焼き直す。

知っておくこと:

- 揺らせるのはカメラ。ボーンやモーフの動きに揺れを足すツールではない。
- VMD にカメラ以外のデータが入っていても、加工するのはカメラキーだけ。
- 視野角が変わる区間では、その区間の元のカメラキーを残し、手ぶれキーを焼き込まない。VMD の視野角は整数度で保存されるため、密なキーにすると各フレームで丸めが入り、ズームが段階的になりやすい。
- 手ぶれを足すのは、カメラの動きを作り終えた最後にする。先に焼き込むとキーが増えて、あとからの手直しが難しくなる。
- 長い VMD では、処理に時間がかかることがある。

詳しい使い方とオプションは [shakevmd/README.md](shakevmd/README.md) にまとめている。

## 開発者向け

### リポジトリ構成

| パス | 内容 |
|---|---|
| [pyproject.toml](pyproject.toml) | 公開コマンド、依存関係、テスト対象の設定 |
| `docs/specs/` | ツールに依存しない仕様・参照資料 |
| `mmd_toolbox/` | VMD 入出力・補間・カメラ座標変換などの共通ライブラリ |
| `shakevmd/` | `shakevmd` コマンド本体と仕様・テスト |
| `sparsevmd/` | `sparsevmd` コマンド本体と仕様・テスト |

### 開発環境

本体ライブラリと CLI のビルド・テストに必要な最小構成。実行時依存は `numpy` と `scipy`(`scipy` は補間曲線フィットで使用。shakevmd の既定の滑らか出力と sparsevmd が利用する)。いずれも pip が各 OS 向けホイールを導入する。

| ツール | 用途 | 備考 |
|---|---|---|
| Python 3.11 以上 | 実装・テスト実行 | Windows は既定の `python` が 3.11 未満のことがあるため `py -3` を使う |
| Git | バージョン管理 | Windows は Git for Windows(Git Bash 同梱)を推奨 |

`numpy`・`scipy`(実行時依存)と `pytest`(開発依存)は `pip install -e ".[dev]"` で導入される。

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
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
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

### テスト実行

```sh
pytest
pytest <パッケージまたはツールのディレクトリ>
```

例:

```sh
pytest shakevmd
```

テストは外部サービス・ネットワーク・MMD本体を必要としない。

Windows でシンボリックリンク作成権限(開発者モードまたは管理者)が無い場合、
上書きガードの symlink テストは自動的に skip される(失敗にはならない)。

### MMD産テストデータについて

一部のテストはMMD本体で作成したVMDファイルを使用する。これらはリポジトリに
コミット済みで、通常は作成・配置の必要はない(`pytest` をそのまま実行できる)。
データを作り直す場合の手順(必要なファイルと条件)は
[mmd_toolbox/tests/data/README.md](mmd_toolbox/tests/data/README.md) を参照。
