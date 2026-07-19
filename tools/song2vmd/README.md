# song2vmd

日本語の歌の音声ファイル(BGM込み可)から、ボーカル分離・音素認識・声の強弱解析を経て、
リップモーション VMD を1コマンドで自動生成するツール。

ボーカル分離・音素認識・音量解析などの外部ツールはすべて内部で呼び出すため、利用者が個別に実行する
必要はない。歌詞テキストの入力は不要で、書き起こしテキストも出力しない。

## インストール

`song2vmd` は音声認識・ボーカル分離に torch/transformers 等の追加依存を使う。トップ [README.md](../../README.md)
の「4. 初回準備コマンドを実行する」の `pip install .` だけでは `song2vmd` コマンド自体は入るが、
これらの依存(`soundfile` を含む)が入らないため、`song2vmd --help` を含めコマンドを呼び出した
直後に `ModuleNotFoundError` で失敗する。`song2vmd` を使う場合は、代わりに次を実行する。

```sh
pip install ".[vocal-analysis]"
```

## 初回実行時のモデルダウンロード

`pip install` の時点ではモデルは取得されない。**`song2vmd` を初めて実行したとき**だけ、内容認識・
音素アライメント・ボーカル分離の学習済みモデルをインターネットから自動取得する(既定構成、
オプション未指定時の実測値)。

| 取得対象 | 用途 | 容量 |
|---|---|---:|
| `openai/whisper-medium` | 内容認識(既定モデル) | 約6.12GB |
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | 音素モデル(強制アライメント) | 約1.26GB |
| Demucs `htdemucs_ft` | ボーカル分離 | 約337MB |
| CMUdict | 英語未知語カタカナ化(既定 `arpakana`) | 約3.8MB |
| 合計 | | 約7.72GB |

`--recognizer-model-id` で既定と異なるモデルを指定した場合や `--english-oov-katakana-method
tinyllama-katakana-converter` を指定した場合は、指定したモデルの容量ぶんが初回実行時に別途追加で
かかる。取得後は各ライブラリの標準キャッシュに保存され、2回目以降の実行では再ダウンロードしない。
内容認識モデル・音素モデルの取得中は、進捗が画面(標準エラー出力)に表示される。

例(既定の `openai/whisper-medium` の代わりに別モデルを使う場合):

```sh
song2vmd <入力ファイル名>.wav --recognizer-model-id sbintuitions/kana-whisper
```

リビジョンを固定したい場合は `--recognizer-model-revision` を組み合わせる。

## 使い方

基本の形:

```sh
song2vmd <入力ファイル名>.wav [オプション]
```

例:

```sh
song2vmd <入力ファイル名>.wav -o <出力ファイル名>.vmd --style powerful
```

出力先を指定しなければ、入力した音声ファイルと同じフォルダに `<入力ファイル名>.vmd` が作られる。

使えるオプションの一覧と短い説明は、`song2vmd --help` でいつでも確認できる。この README は主な
オプションと使いどころをまとめたもの。

## 歌い方スタイルプリセット

`--style` で、歌い方に合わせて口の開き量やタイミングの既定値をまとめて切り替えられる。

| 名前 | 想定する歌い方 |
|---|---|
| `pop`(既定) | 標準的・はっきりした歌唱 |
| `ballad` | しっとり・落ち着いた歌唱 |
| `powerful` | 力強い・ロック・シャウト |
| `whisper` | ささやき・静かな歌唱 |
| `rap` | 早口・テンポの速い歌唱 |

プリセットは出発点で、`--open-max` などの個別オプション(下記)で上書きできる。

## 注意事項

- 対象は日本語の歌のみ。母音認識の誤り(歌唱の崩れ・ロングトーン・無声化・コーラス混入など)はそのまま口形に出る。
- 既定で生成するのはあ・い・う・え・おの標準口モーフのみ。撥音は既定で閉口、`--n-morph` で「ん」モーフに切り替えられる。まばたき・眉・感情表現などの表情モーフは生成しない。日本語以外の言語、歌詞テキスト指定による高精度化には対応しない。
- 対応する音声ファイル形式は WAV・FLAC・OGG・mp3。ffmpeg を OS にインストールしておくと、mp4/aac
  など追加の形式にも対応する。
- 長い曲は無音区間で自動分割して処理する(`--max-duration`、既定300秒。VMDのフレーム番号上限対策
  ではなく処理資源対策)。
- GPU(CUDA)があれば自動的に使う(`--device auto`)。VRAM が不足すると大幅に遅くなり、警告が表示
  される。その場合は `--device cpu` を指定すると改善することがある。
- PC の性能(GPU の有無・CPU 性能など)によっては、処理に時間がかかることがある。
- 出力先パスに既存ファイルがある場合、`--overwrite` を付けない限り上書きしない。
- 処理中に GPU メモリ超過やスワップの発生を実際に観測した場合、標準エラーへ警告を出す(処理内容や
  出力は変えない)。

## オプション

### 入出力

| オプション | 既定値 | 値と効き方 |
|---|---|---|
| `<入力ファイル名>` | 必須 | 入力する音声ファイル(wav/mp3 等)。 |
| `-o`, `--output PATH` | `<入力ファイル名>.vmd` | 出力先。指定しない場合は、入力ファイルの名前に `.vmd` を付けた VMD を作る。 |
| `--overwrite` | 無効 | 出力先に既にファイルがある場合の上書きを許可する。指定しない場合、出力先に既存ファイルがあると止まる。 |
| `--model-name NAME` | `song2vmd <版>` | VMD に格納するモデル名(最大20バイト・Shift-JIS)。 |

### リップモーションの効かせ方

| オプション | 既定値 | 値と効き方 |
|---|---|---|
| `--style NAME` | `pop` | 歌い方スタイルプリセット(上表)。開き量レンジ・タイミングをまとめて切り替える。 |
| `--vowel-gain a:i:u:e:o` | `1:1:1:1:1` | 母音別の開き量微調整倍率。プリセットの母音別倍率へ要素ごとに乗算する。 |
| `--open-max V` | プリセット値 | 口の開き量の上限(0.0〜1.0)。開けすぎを防ぐ。 |
| `--intensity-curve P` | `0.6` | 強弱→開き量の非線形指数(累乗則)。弱い声を持ち上げすぎないようにする。 |
| `--coarticulation FRAMES` | プリセット値 | 隣接母音の協調調音の重なり長の上限(フレーム)。 |
| `--anticipation FRAMES` | プリセット値 | 母音口形を音より先行させる最大フレーム数。 |
| `--min-hold FRAMES` | プリセット値 | 最小保持フレーム。これより短いモーラは併合・間引きする。 |
| `--silence-threshold ON:OFF` | `0.06:0.10` | 無音(閉口)判定のヒステリシス開始・終了しきい値。 |
| `--n-morph`, `--no-n-morph` | `--no-n-morph` | 「ん」モーフを使うかどうか。既定は閉口。「ん」モーフを使いたいモデルでは `--n-morph` を指定する。 |

### 音声前段(バックエンド選択)

| オプション | 既定値 | 値と効き方 |
|---|---|---|
| `--separate-vocals MODE` | `always` | ボーカル分離の実施方針。`always`(常に分離する) / `never`(分離しない) から選ぶ。分離自体を行わない場合は `never` を指定する。 |
| `--separator NAME` | `audio-separator-htdemucs-ft` | ボーカル分離バックエンドの選択。 |
| `--recognizer-model-id ID` | (未指定) | 内容認識モデルの指定。未指定時は既定モデル(`openai/whisper-medium`)を使う。 |
| `--recognizer-model-revision REV` | (未指定) | `--recognizer-model-id` と組で使うリビジョン指定。`--recognizer-model-id` を指定してこちらを省略すると、そのモデルの最新リビジョンを使う。 |
| `--recognizer-retry`, `--no-recognizer-retry` | `--recognizer-retry` | 内容認識のトリガ式リトライ(エコー幻覚・反復幻覚対策)の有効・無効。 |
| `--forced-aligner NAME` | `wav2vec2-ctc-forcedalign` | 強制アライメント段のバックエンド選択。既定のままなら以下の `--sofa-*` は不要。 |
| `--sofa-python PATH` | なし | SOFA 専用venvのPython実行ファイルパス。`--forced-aligner sofa-forcedalign` 選択時のみ必須。 |
| `--sofa-root PATH` | なし | SOFA リポジトリのルートパス。`--forced-aligner sofa-forcedalign` 選択時のみ必須。 |
| `--sofa-checkpoint PATH` | なし | SOFA チェックポイント(`.ckpt`)ファイルパス。`--forced-aligner sofa-forcedalign` 選択時のみ必須。 |
| `--sofa-timeout SEC` | `300` | SOFA サブプロセス1回あたりのタイムアウト秒数。 |
| `--english-oov-katakana-method NAME` | `arpakana` | 英語未知語カタカナ化フォールバックの変換方式。`arpakana`(GPU不要)または `tinyllama-katakana-converter`(生成モデル使用)。 |

### 実行環境と長尺分割

| オプション | 既定値 | 値と効き方 |
|---|---|---|
| `--device MODE` | `auto` | 実行デバイスの選択。`auto` は GPU(CUDA)が利用可能なら GPU を使う。`cpu` は GPU を使わない(VRAM 不足環境の回避手段)。 |
| `--max-duration SEC` | `300` | 長尺の自動分割境界。`0` で無効。 |

### 診断と出力確認

| オプション | 既定値 | 値と効き方 |
|---|---|---|
| `--dry-run` | 無効 | 出力せず、選択したバックエンド・検出モーラ数・生成キー数などの処理計画と診断を表示する。 |
| `--keep-intermediate` | 無効 | 中間生成物(正規化PCM・分離後ボーカルWAV・認識結果)を残す(診断用)。 |
| `-v`, `--verbose` | 無効 | 通常実行でも `--dry-run` と同じ診断レポートを標準出力へ表示する(出力VMDは書く)。 |
| `--quiet` | 無効 | 進捗表示を抑制する。警告・診断・終了コードは抑制しない。 |
| `--help` | — | オプションの一覧と短い説明を表示して終了する。 |
| `--version` | — | バージョンを表示して終了する。 |

### 機械利用(自動化・ツール連携)

他のソフトウェアから子プロセスとして呼び出して使うためのオプション。手作業で使うときは要らない。

| オプション | 既定値 | 値と効き方 |
|---|---|---|
| `--machine` | 無効 | 処理を実行するとき、画面(標準出力)への表示を1行1件の JSON(進捗・警告・結果・エラー)に切り替える。出力する VMD と終了コードは変わらない。 |
| `--describe` | 無効 | 音声を読まずに、受け付けるオプションとプリセットの一覧を JSON で出力して終了する。 |

出力する JSON の形など、機械利用の詳しい仕様は [song2vmd.md](song2vmd.md) を参照。

## SOFA を使う場合のセットアップ(上級者向け)

`--forced-aligner sofa-forcedalign` で音素タイミング合わせを SOFA(Singing-Oriented Forced Aligner)に
置き換えたい場合のセットアップ。**既定のままなら本節は一切不要。**

- SOFA 本体は MIT ライセンスだが、**学習済みモデル(チェックポイント)は別ライセンス**。現在入手できる
  日本語チェックポイントは、商用利用ができないもの(Greenleaf2001/SOFA_Models)や、商用利用に個別相談が
  必要な学習データを含むもの(colstone/SOFA_Models)である。**商用作品に使う場合は、使うチェックポイント
  のライセンスを必ず自分で確認すること。** この理由から `song2vmd` はチェックポイントを同梱せず、既定でも
  使わない。
- SOFA は `song2vmd` 本体とは別の Python 環境(専用 venv)で動かす。SOFA が要求するライブラリの版が
  `song2vmd` 本体のものと両立しないためで、`song2vmd` が SOFA をサブプロセスとして呼び出す。
- **SOFA は非ASCII文字(日本語など)を含むパスを扱えない**。SOFA 本体の置き場所・チェックポイントの
  置き場所は ASCII のみのパスにする(下記手順の `C:\tools\...` はその一例)。加えて `song2vmd` が
  内部で使う一時ディレクトリ(既定では OS の一時フォルダ)にも同じ制約がかかるため、Windows の
  ユーザー名に日本語などの非ASCII文字が含まれる場合は、実行前に環境変数 `TEMP`・`TMP` を ASCII のみの
  パス(例: `C:\temp`)に変更しておく。

### 必要なもの

| もの | 条件 |
|---|---|
| Python 3.10 | SOFA の依存ライブラリ(numpy 1.24 系など)が新しい Python では導入できないため、3.12 以上を使う `song2vmd` 本体とは別に 3.10 を入れる |
| SOFA 本体 | <https://github.com/qiuqiao/SOFA> |
| チェックポイント(`.ckpt` ファイル) | 利用者が配布元から入手する(上記ライセンス注意)。大きさは配布元により約53MB〜約1.2GB |
| (任意)NVIDIA GPU | 無くても動くが、あると速い |

### 手順(Windows)

1. **Python 3.10 をインストールする**(インストール済みなら飛ばす)。[Python 公式ダウンロードページ](https://www.python.org/downloads/) から 3.10 系を入れる。

2. **SOFA 本体を取得する**。PowerShell で:

   ```powershell
   git clone https://github.com/qiuqiao/SOFA C:\tools\SOFA
   ```

   git を使わない場合は GitHub の `Code` → `Download ZIP` で取得し、`C:\tools\SOFA` へ展開する(置き場所は
   ASCII のみのパスであればどこでもよい。以下は `C:\tools\SOFA` に置いた例で書く)。

3. **SOFA 専用の venv を作る**:

   ```powershell
   py -3.10 -m venv C:\tools\sofa_venv
   C:\tools\sofa_venv\Scripts\Activate.ps1
   ```

4. **PyTorch を入れる**(SOFA の依存一覧は torch を含まないため先に手で入れる)。

   GPU(NVIDIA)がある場合:

   ```powershell
   pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

   GPU が無い場合:

   ```powershell
   pip install torch==2.5.1 torchaudio==2.5.1
   ```

5. **SOFA の依存ライブラリを入れる**:

   ```powershell
   pip install -r C:\tools\SOFA\requirements.txt
   ```

6. **チェックポイントを入手する**。配布元(GitHub 上の Greenleaf2001/SOFA_Models、colstone/SOFA_Models
   など)から日本語向けの `.ckpt` ファイルをダウンロードし、ASCII のみのパス(例:
   `C:\tools\sofa_ckpt\japanese.ckpt`)へ置く。**ダウンロード前に配布元のライセンス条件を確認すること**
   (上記の注意)。

7. **`song2vmd` から使う**。SOFA 専用 venv を抜け、`song2vmd` 本体の venv(トップ [README.md](../../README.md)
   の「5. ツールを実行する」で有効にしたもの)に切り替える:

   ```powershell
   deactivate
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   ```

   切り替えたら次を実行する:

   ```powershell
   song2vmd 歌.mp3 --forced-aligner sofa-forcedalign --sofa-python C:\tools\sofa_venv\Scripts\python.exe --sofa-root C:\tools\SOFA --sofa-checkpoint C:\tools\sofa_ckpt\japanese.ckpt
   ```

   3つのパス指定(`--sofa-python`・`--sofa-root`・`--sofa-checkpoint`)は SOFA 使用時は毎回必須。SOFA の
   1回の呼び出しが300秒を超えて打ち切られる場合は `--sofa-timeout` で延ばせる。

### 動作確認済みの組み合わせ

| 項目 | 版 |
|---|---|
| Python(SOFA 専用 venv) | 3.10.11 |
| torch / torchaudio | 2.5.1(CUDA 12.1 版) |
| numpy | 1.24.4 |
| librosa | 0.9.2 |

既定構成(SOFA を使わない)でもリップモーションは生成できる。実曲の聴き比べでは既定構成の方が自然に
見える場合があり、SOFA は「試せる選択肢」という位置づけ。まず既定で生成し、結果に不満があるときに
SOFA を試すのがよい。SOFA 用の容量の目安は、本体コードが約5MB、専用venvが約5.2GB(torchを含む)、
チェックポイントが約53MB〜約1.2GB。

## 進捗表示

ボーカル分離・音素認識・長尺分割の各チャンク処理・モーフ生成は数分かかることがあり、その間ターミナルへ
何も出ないと止まっているのか分からない。処理の段階をターミナル(標準エラー出力)へライブ表示する。
この表示が出るのはターミナルへそのまま実行した時だけで、出力をファイルやパイプへ渡した時や `--quiet`
指定時は出さない。モデルの初回取得がネットワークダウンロードを要した区間は、取得中のモデルと割合も
併せて表示する。

さらに細かい仕様を確認したい場合は [song2vmd.md](song2vmd.md) を参照。
