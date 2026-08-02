# mmd-toolbox

MikuMikuDance (MMD) 向けのユーティリティツール群。

## ツール

| ツール | 概要 | バージョン | 詳しい使い方 |
|---|---|---|---|
| `mocapvmd` | モーションキャプチャー由来のボーンモーション VMD を最適化する | 0.0.1 | [tools/mocapvmd/README.md](tools/mocapvmd/README.md) |
| `shakevmd` | カメラモーション VMD ファイルに手ぶれ効果を追加する | 0.0.2 | [tools/shakevmd/README.md](tools/shakevmd/README.md) |
| `song2vmd` | 日本語の歌の音声ファイルから、リップモーション VMD を自動生成する | 0.0.1 | [tools/song2vmd/README.md](tools/song2vmd/README.md) |

## 使い方

### 1. Python 3.12 以上をインストールする

- [Python 公式ダウンロードページ](https://www.python.org/downloads/)
- [Python 公式ドキュメント: Python のセットアップと利用](https://docs.python.org/ja/3/using/index.html)

Python のインストール手順は AI に次のように聞いてね。

```text
WindowsまたはmacOSでPython 3.12以上をインストールして、ターミナルでpythonコマンドが使えるようにする手順を、初心者向けに教えて。
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

### 3. [README.md](README.md) が入っているフォルダへ移動する

Windows では PowerShell、macOS ではターミナルを開き、この README が入っているフォルダへ移動する。

1. PowerShell またはターミナルに `cd ` と入力する。`cd` の後ろには半角スペースを入れる。
2. 展開したフォルダを開き、[`README.md`](README.md) が見えるフォルダを PowerShell またはターミナルへドラッグ＆ドロップする。
3. Enter キーを押す。

Enter キーを押す前は、たとえば次のようになる。

```powershell
cd "C:\Users\ユーザー名\Downloads\mmd-toolbox-main\mmd-toolbox-main"
```

以降のコマンドは、このフォルダの中で実行する。

### 4. 仮想環境を作成する

Windows では PowerShell で、次のコマンドを上から順番に実行する。

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

### 5. ツールを実行する

ツールは、仮想環境(.venv)を有効にしたウィンドウで実行する。[手順4](#4-仮想環境を作成する)をした直後の同じウィンドウなら、すでに有効なのでそのまま実行できる。

ウィンドウを開き直したときは、[手順3](#3-readmemd-が入っているフォルダへ移動する)と同じようにフォルダへ移動してから仮想環境を有効にする。Windows では PowerShell で次を実行する。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

macOS では次を実行する。

```sh
source .venv/bin/activate
```

有効にしたら、各ツールのコマンドを実行する。入力ファイルは、コマンドのうしろにパスを書く。ファイルを PowerShell やターミナルへドラッグ＆ドロップすると、パスが入る。

## mocapvmd: モーションキャプチャーのモーションを最適化する

`mocapvmd` は、モーションキャプチャー由来のボーンモーション VMD を最適化するツール。
細かいノイズ・足の接地中の横滑りを抑え、キーフレームを圧縮した扱いやすい VMD を作る。

使い方:

1. モーションキャプチャーで作成した VMD ファイルを用意する。
2. 次のコマンドを実行する。

```sh
mocapvmd <入力ファイル名>.vmd
```

最適化した新しい VMD が、入力した VMD と同じフォルダに `<入力ファイル名>_mocap.vmd` という名前で作られる。

このツールは処理が重いので、時間がかかる場合はプリセットに `faster` を指定する。
キーが大きく減るぶん、細かい動きが省かれて元の動きとの差が大きくなる。PC の性能や VMD の長さによっては、`faster` でも時間がかかる。

```sh
mocapvmd <入力ファイル名>.vmd --preset faster
```

処理の特徴:

- キーフレームを圧縮するときは元の動きとのずれを決めた範囲に収め、間のキーフレームを省いて残したキーの間をなめらかな曲線(ベジェ)でつなぐ。MMD で手で編集できる量のキーフレームになる。

注意事項:

- 既定では足の接地中の横滑りを除去する。意図的に足を滑らせる動きでは、オプション `--foot-slide-suppression` を下げると横滑りが残る。
- このツールは元の動きに手を加えるので、出力は元と完全に同じにはならない(とくに細部と接地中の足元の動きは変わる)。ただし大きなステップ・ジャンプ・ターン・手振りなどの動きのアクセントは保護し、全体のタイミングや重心移動の流れは保つようにしている。

詳しい使い方とオプションは [tools/mocapvmd/README.md](tools/mocapvmd/README.md)

## shakevmd: カメラに手ぶれ効果を追加する

`shakevmd` は、MMD のカメラモーション VMD ファイルに手ぶれ効果を追加するツール。
VMD は、MMD でモーションやカメラの動きを保存するファイル形式。ここで扱うのはカメラの動きが入った VMD。

使い方:

1. カメラモーション VMD ファイルを用意する。
   - MMD で `編集` → `カメラフレーム全て選択` を選んでから、`ファイル` → `モーションデータ保存` で VMD を保存する。
2. 次のコマンドを実行する。

```sh
shakevmd <入力ファイル名>.vmd
```

揺れを足した新しい VMD が、入力した VMD と同じフォルダに `<入力ファイル名>_shake.vmd` という名前で作られる。

処理の特徴:

- 元のカメラの動きを1フレームごとに読み取り、帯域制限ノイズから作った回転・位置オフセットを加えてカメラキーとして焼き直す。

注意事項:

- 揺らせるのはカメラ。ボーンやモーフの動きに揺れを足すツールではない。
- VMD にカメラ以外のデータが入っていても、加工するのはカメラキーだけ。
- 視野角が変化する区間には手ぶれを焼き込まず、元のカメラキーを残す。視野角は整数でしか保存できず、キーを増やすとズームがカクつくため。
- 手ぶれを足すのは、カメラの動きを作り終えた最後にする。先に焼き込むとキーが増えて、あとからの手直しが難しくなる。
- 長い VMD では、処理に時間がかかることがある。

詳しい使い方とオプションは [tools/shakevmd/README.md](tools/shakevmd/README.md)

## song2vmd: 歌声からリップモーション VMD を自動生成する

`song2vmd` は、日本語の歌の音声ファイル(BGM込み可)から、ボーカル分離・音素認識・音量解析を経て、あ・い・う・え・お・(ん)によるリップモーション VMD を自動生成するツール。

追加のインストール:

1. NVIDIA GPU を搭載している場合は [PyTorch 公式サイト](https://pytorch.org/get-started/locally/) の `Compute Platform` で CUDA のバージョンを選び、次のコマンドの `--index-url` をそれに合わせて実行する(次は CUDA 12.6 の場合)。

   ```sh
   pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
   ```

2. 次を実行する(song2vmd が使う音声認識・ボーカル分離のライブラリを追加する)。

   ```sh
   pip install ".[vocal-analysis]"
   ```

使い方:

1. 歌の音声ファイル(wav・mp3 等)を用意する。
2. 次のコマンドを実行する。

```sh
song2vmd <入力ファイル名>.wav
```

生成したリップモーション VMD が、入力した音声ファイルと同じフォルダに `<入力ファイル名>.vmd` という名前で作られる。

処理の特徴:

- ボーカル分離・音素認識・音量解析は内部で行い、外部ツールを利用者が個別に実行する必要はない。
- 声の強弱を各モーラの口の開き量に反映し、モーラごとに口形を保持する、アニメ的にはっきり開閉するリップモーションを作る。

注意事項:

- 対象は日本語の歌のみ。母音認識の誤り(歌唱の崩れ・ロングトーンなど)はそのまま口形に出る。
- 生成するのはあ・い・う・え・お・(ん)の標準口モーフのみ。「ん」はオプション。
- 感情・表情モーフ(まばたき・眉など)の生成、日本語以外の言語、歌詞テキスト指定による高精度化には対応しない。
- 対応する音声ファイル形式は WAV・FLAC・OGG・mp3。ffmpeg がインストール済みの環境であれば、mp4/aac など追加の形式にも対応する。
- 初回実行時だけ、内容認識・音素アライメント・ボーカル分離の学習済みモデルをインターネットから自動取得する(既定構成の合計は約7.7GB)。取得後はキャッシュされるため2回目以降のダウンロードは発生しない。
- GPU(CUDA)は CUDA 版の `torch` を入れてあれば自動的に使う。VRAM が不足すると大幅に遅くなり、警告が表示される。その場合は `--device cpu` を指定すると改善することがある。
- PC の性能(GPU の有無・CPU 性能など)によっては、処理に時間がかかることがある。

詳しい使い方とオプションは [tools/song2vmd/README.md](tools/song2vmd/README.md)

## 開発者向け

### リポジトリ構成

| パス | 内容 |
|---|---|
| [pyproject.toml](pyproject.toml) | 公開コマンド、依存関係、テスト対象の設定 |
| `docs/specs/` | ツールに依存しない仕様・参照資料 |
| `docs/conventions/` | 層タクソノミー・バージョン付けなどツール横断の規約 |
| `scripts/` | CI と開発者の双方から実行するリポジトリ保守スクリプト |
| `libs/cli_events/` | 機械モードのイベント送出の共有ドメインライブラリ |
| `libs/cli_options/` | 引数検証と自己記述の制約公開を同一定義から導く共有ドメインライブラリ |
| `libs/cli_progress/` | 進捗ライブ表示の共有ドメインライブラリ |
| `libs/cli_progress_router/` | 進捗をイベント送出とライブ表示へ振り分ける共有ドメインライブラリ |
| `libs/cli_resource_watch/` | 資源逼迫・実行構成の観測と判定の共有ドメインライブラリ |
| `libs/lipsync/` | リップモーション生成の共有ドメインライブラリ |
| `libs/pmx/` | PMX 読み取り・ボーン階層・FK 評価のライブラリ |
| `libs/vmd/` | VMD 入出力・補間・カメラ座標変換・キーフレーム疎化のライブラリ |
| `libs/vocal_analysis/` | 音声解析の共有ドメインライブラリ |
| `libs/vpr/` | VOCALOID プロジェクトファイル(vpr)の読み書き・休符導出のライブラリ |
| `tools/<ツール>/` | CLI ツール層。各ツールはコマンド本体・仕様書・テストを直下に置く(利用者向けに公開するツールは README も)。公開コマンドの一覧は [pyproject.toml](pyproject.toml)、おもなツールの使い方は冒頭の [ツール](#ツール) 一覧 |

### 開発環境

本体ライブラリと CLI のビルド・テストに必要な最小構成。実行時依存は `numpy` と `scipy`(`scipy` は補間曲線フィットで使用。shakevmd の既定のスムージングと sparsevmd が利用する)。いずれも pip が各 OS 向けホイールを導入する。

| ツール | 用途 | 備考 |
|---|---|---|
| Python 3.12 以上 | 実装・テスト実行 | Windows は既定の `python` が 3.12 未満のことがあるため `py -3` を使う |
| Git | バージョン管理 | Windows は Git for Windows(Git Bash 同梱)を推奨 |
| GitHub CLI(`gh`) | リリース作業(PR 作成・マージ・Release 確認)の実行 | 任意。ツールをリリースするときだけ必要。初回に `gh auth login` で認証する |
| lychee | ドキュメントのリンク検査 | pip では入らないため各自で導入する。設定は [lychee.toml](lychee.toml) |
| ruff | Python コードの静的検査 | 開発依存として導入される。設定は [pyproject.toml](pyproject.toml) |

`numpy`・`scipy`(実行時依存)と `pytest`・`ruff`(開発依存)は `pip install -e ".[dev,vocal-analysis]"` で導入される。
`song2vmd`(音声認識・ボーカル分離を使う)のテスト実行には、追加で `vocal-analysis` extra
(`torch`・`transformers`・`pyopenjtalk-plus`等)が要る。開発環境構築では両方合わせて
`pip install -e ".[dev,vocal-analysis]"` を使う。

### 環境構築

リポジトリルートで仮想環境を作成し、開発インストールする。

#### macOS / Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,vocal-analysis]"
```

#### Windows (PowerShell)

```powershell
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,vocal-analysis]"
```

cmd の場合、有効化のみ `.\.venv\Scripts\activate.bat` に読み替える。

### AI 支援開発ワークフロー(任意)

本リポジトリの開発プロセス(TDD・Codex によるレビューループ・コミット委譲)は Claude Code と Codex を併用する(`.claude/` 配下のスキル・フック)。これらを使う場合は以下も導入する。

| ツール | インストール | 用途 |
|---|---|---|
| Node.js 20 以上(LTS 推奨) | 公式インストーラ / nvm | Codex companion(レビューループ)の実行 |
| Claude Code | `npm install -g @anthropic-ai/claude-code` | AI 開発ハーネス(スキル・フック・レビュー) |
| Codex CLI | `npm install -g @openai/codex` | コードレビューの実行体。**既定モデルに追従するため最新バージョンを推奨**(古いとモデル非対応で API エラーになる) |
| Codex プラグイン | Claude Code 内で `/plugin marketplace add openai/codex-plugin-cc` の後 `/plugin install codex@openai-codex` | Claude Code から Codex を呼ぶ連携(`codex-review-loop` スキルが使用) |

**Windows 追加要件**: Claude Code は Bash ツール・フック・[watchdog.sh](.claude/skills/codex-watchdog/watchdog.sh) の実行に **Git Bash** を使う(Git for Windows 同梱)。Git Bash が無いと Codex レビューループやフックが動作しないため、Windows では Git Bash の導入が必須。

### 検証の実行

変更を確定させる前に通す検査の一覧・コマンド・合格条件は
[docs/conventions/verification.md](docs/conventions/verification.md) が持つ。

そのうちテストと静的検査は、作業中に対象を絞って回せる。次は shakevmd だけに絞る例。

```sh
pytest tools/shakevmd
ruff check tools/shakevmd
```

テストは外部サービス・ネットワーク・MMD本体を必要としない。

Windows でシンボリックリンク作成権限(開発者モードまたは管理者)が無い場合、上書きガードの symlink テストは自動的に skip される(失敗にはならない)。

### MMD産テストデータについて

一部のテストはMMD本体で作成したVMDファイルを使用する。これらはリポジトリにコミット済みで、通常は作成・配置の必要はない(`pytest` をそのまま実行できる)。データを作り直す場合の手順(必要なファイルと条件)は[libs/vmd/tests/data/README.md](libs/vmd/tests/data/README.md) を参照。
