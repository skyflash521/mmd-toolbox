# vocal_analysis 外部ツール検討資料

`vocal_analysis`(音声前段の共有モジュール)が連携する外部ツールの候補を、パイプラインの各ステージごとに
比較し、採用ツールと代替候補を決めるための補助資料。仕様本体(連携機構・正規化中間形式・採用ツールの正本)は
[vocal_analysis.md](vocal_analysis.md) §8・§2 を正とする。採用は実データでの品質・導入性の評価で見直しうる
(見直す場合は正本を先に更新する)。

## 0. 前提と評価軸

差し替え可能な委譲ステージ([vocal_analysis.md](vocal_analysis.md) §2): S1 ボーカル抽出 / S2 音素・母音認識。
S0 入力読み込みは固定の内部処理(soundfile/ffmpeg)で、差し替えアダプタの対象ではない。強弱解析(RMS)は
`vocal_analysis` 中核(numpy/scipy)で行い、外部ツールに依存しない。

設計上の前提:

- **利用者に外部コマンドを叩かせない**。利用者はCLIの1コマンドで完結し、外部ツールは
  `vocal_analysis` が内部で呼ぶ。呼び出しは **ライブラリAPI(Python)を優先**し、無いものだけ内部管理の
  サブプロセスで呼ぶ。依存は `vocal_analysis` 側に閉じ、本リポジトリ本体の必須依存は `numpy/scipy` のまま
  保つ。
- **歌詞の書き起こし(意味のあるテキスト化)はしない**。後段に必要なのは「母音・音素の時刻」であり単語では
  ない。歌唱の自動書き起こしは誤りが多く伝播するため、音声から母音・音素を直接認識する
  ([vocal_analysis.md](vocal_analysis.md) §5)。
- 本リポジトリの導入手順は Windows/macOS を対象とする([README.md](../../README.md))。ライブラリで導入
  できること・Windows での導入容易性を重視する。

評価軸: 日本語適性 / 品質 / 速度・GPU要否 / ライブラリ提供と導入容易性(Windows) / ライセンス /
決定論 / 出力の解析しやすさ / 保守状況。

---

## 1. S1 ボーカル抽出

BGM込み音源からボーカルWAVを得る(S1はステレオ原音入力が高品質。S2向け 16kHz mono 化はS1後)。ボーカル品質は
S2の母音認識精度に直結する。ライブラリAPIで in-process 呼び出しできることを重視する。

| ツール | Python API | 品質 | 速度・要件 | 導入 | ライセンス | 備考 |
|---|---|---|---|---|---|---|
| **Demucs v4 (htdemucs)** 生API | `demucs.api.Separator` | 高(SDR≈9dB) | 中。CPU可/GPUで速い | pip。ただし依存 `torchaudio` が `<2.2` 固定で新しい Python(3.13等)向けビルドが無く導入不能 | MIT | 元repoは保守終了(archive)。限定bugfix fork: adefossez/demucs |
| **audio-separator** (UVR系) | `Separator` クラス | モデル次第で最高峰(同梱 Demucs v4 htdemucs_ft はモデル一覧上のボーカルSDR≈10.8) | モデル次第。ONNX/torch | pip。容易。クロスプラットフォーム。`torchaudio` 上限に縛られない | MIT | UVRのMDX-Net/VR/Demucs/MDXCを切替。Demucs系モデルも `demucs_params.shifts` で非決定要素を制御可 |
| **Spleeter** | あり(TF) | 中(やや古い) | 高速・軽量 | pip(TensorFlow依存) | MIT | 速いが品質は上2者に劣る |

**採用: audio-separator 経由の Demucs v4 htdemucs_ft**(`audio_separator.separator.Separator`)。理由:
品質・MIT・Python APIでin-process呼び出し可・GPU不要でも動作という Demucs の採用理由をそのまま満たし、
かつモデル実体(Demucs v4 の重み)も変えない。生 `demucs.api`(adefossez fork)は依存 `torchaudio` を
`<2.2` に固定しており、この上限を満たす `torchaudio` ビルドが無い新しい Python では導入できないため
採用しない。同じ Demucs v4 の重みを audio-separator 経由で実行する。再現性のため
`demucs_params={"shifts": 0, ...}` で非決定要素を固定する。分離不要なボーカル単体入力は Separator 抽象の
`mode=never` で扱う([vocal_analysis.md](vocal_analysis.md) §8.1)。

ボーカル抽出ツールの切り替えは [vocal_analysis.md](vocal_analysis.md) §8.1 の **Separator 抽象**の背後で行う。
Separatorは出力に「ボーカルWAVのパス」だけを約束し、内部のライブラリ・モデル・トラック構成は各実装に閉じる。

---

## 2. S2 音素・母音認識(文字起こしを使わない)

### 2.1 なぜ文字起こし(ASR)を使わないか

- **歌唱ASRは誤りが多い**。研究では同一歌詞で歌唱WER≈0.56 / 朗読0.14(約4倍悪化)。伴奏より歌い方の影響が
  大きく、ハルシネーション(無発話区間の捏造)や非語彙発声(ラララ等)に弱い。
- **そもそも単語は不要**。後段に必要なのは「時刻ごとの母音(口形)・音素」。日本語の5母音は音響的に明瞭で、
  歌唱では母音スペクトルがむしろ安定し、音声から直接認識しやすい。
- **フォースアライメントは正しい歌詞が前提**(与えた文字列に音声を合わせる)ため、歌唱では前提が崩れる。
  一方 **wav2vec2-CTC は文字列なしで音素+時刻を出せる**。Julius も phone-loop(文法なし音素認識)を構成すれば
  文字列なしで音素+時刻を出せる。

→ 文字起こし+G2P+フォースアライメントの多段構成は採らず、**文字列なしの音素・母音認識1段**(S2)とする。

### 2.2 候補

| ツール | 呼び出し | 日本語/汎用 | 時刻精度 | 速度・要件 | ライセンス | 歌唱頑健性 |
|---|---|---|---|---|---|---|
| **wav2vec2 音素認識** | transformers(in-process) | 多言語/日本語 | CTC近似(補正で実用) | CPU可(GPUで速)。torchはDemucsと共有(transformers本体・モデル取得は新規) | transformers=Apache / torch=BSD + 許諾モデル | ◎ 自己教師ありで歌唱に汎化 |
| **Julius 音素認識** | C実行ファイル(内部subprocess) | 日本語(無償音響モデル) | フレーム単位(高) | 軽い。別系統の追加 | エンジン=修正BSD(許諾的) | △ speech-HMMで歌唱は域外 |
| ~~Allosaurus~~ | Python API | 汎用 | 近似 | 軽い | **GPL-3.0 → MIT本体と非互換で不可** | ◎ |

> 補足: CTC系の時刻はトークン単位の粗いオフセットで、音素境界そのものではない。母音区間の境界の精緻化
> (近傍にRMSオンセットがあるときそれへ寄せる)は利用先(口パク生成系の入口)が行う
> ([vocal_analysis.md](vocal_analysis.md) §5)。認識器は5母音と子音、未割当(gap)を区別できれば足り(無音/閉口の
> 確定は利用先がRMS併用で行い、両唇閉鎖は音素から判定)、語彙認識より要件は緩い。

**採用: wav2vec2 音素認識(transformers + 許諾モデル)**。決め手:

1. **歌唱頑健性**(本方式の品質の要)で、自己教師あり(SSL)が speech-HMMの Julius に勝る。
2. **純Pythonでin-process**に呼べ、「利用者にコマンドを叩かせない/ライブラリ呼び出し」方針に最も合う。
3. **torch を Demucs と共有できる**(Julius は C/モデル/jconf の別系統を追加することになる)。ただし
   transformers 本体とモデル取得(ダウンロード・キャッシュ・メモリ・初回ネットワーク)は新規に必要になる。
   モデルの版固定・キャッシュ先・オフライン挙動は [vocal_analysis.md](vocal_analysis.md) §5.1・§8.3 が定める。
4. ライセンスが清浄。transformers=Apache-2.0 / torch=BSD-3 に加え、音素モデル
   `facebook/wav2vec2-lv-60-espeak-cv-ft` は **Apache-2.0(確認済み)**。多言語eSpeak音素を出力し、
   IPA音素→あいうえお母音へ写像する。

弱点の時刻の粗さは、トークン境界を一次情報とし利用先がRMSオンセットで精緻化して実用化する。**Julius は
「高精度な時刻」の代替**として残す(ただし文字列なし運用には phone-loop の音響モデル・辞書・設定の構成が
必要)。歌唱品質か速度が問題になれば評価する。**Allosaurus は GPL-3.0(LICENSE実物が GNU GPL v3)で
MIT本体と非互換のため不採用**。

S2はこの一段で後段の成否が決まるため、採用の確定は **S-1 認識ゲート**に従う: 代表となる日本語歌唱サンプルで
母音正解率・境界時刻ずれを測り、あらかじめ定めた受入基準を満たすことを確認してから採用を確定する
([vocal_analysis.md](vocal_analysis.md) §9)。

---

## 3. S0 入力読み込み(ffmpeg の扱い)

**結論: ffmpegは soundfile で読めない形式のために要るが、範囲は限定的。いずれにせよ本リポジトリには
同梱・再配布しない**。

必要な理由:

- 入力には mp3/m4a/aac 等があり、復号と、各バックエンドが要求する 16kHz mono への変換が要る(Julius=
  16k/16bit/mono、wav2vec2=16k)。
- `soundfile` は、現行の同梱 libsndfile(1.1.0 以降)で WAV/FLAC/OGG に加え **mp3 も復号できる**。よって多くの
  入力は soundfile で読め、再サンプリングは numpy/scipy でも可能。**soundfile で読めない形式**(例: 一部の
  m4a/aac、コンテナ依存のもの)のために ffmpeg が要る。

ライセンスの前提:

- ffmpeg は LGPL/GPL。**ビルド済みバイナリを自分のリポジトリにコミット/自分のパッケージに同梱して配ると、
  再配布の義務が我々に生じる**(GPLビルドは特に問題)。よってバイナリの再配布はしない。

**利用者環境の ffmpeg を実行時に自動検出(PATH/既知パス)して内部で呼ぶ(確定)**。我々は ffmpeg バイナリを
一切配らないため、ffmpeg ビルドのライセンス(LGPL/GPL)は我々の義務にならない。利用者は ffmpeg を一度だけ
導入すればよい。`soundfile` で読める形式(WAV/FLAC/OGG/mp3 等)は soundfile で読み、読めない形式が無ければ
ffmpeg 自体が不要なことも多い。

別案として `imageio-ffmpeg` 等でffmpegを同梱配布するpipパッケージに依存すれば導入は楽になるが、同梱
バイナリのビルド・ライセンスがパッケージ依存になるため**採らない**。いずれにせよ利用者に ffmpeg コマンドは
叩かせない。本リポジトリは MIT([../../LICENSE](../../LICENSE))。

---

## 4. 採用ツールと選定理由

各ステージの採用ツールは1つで、既定アダプタとして用いる(採用ツールの正本は
[vocal_analysis.md](vocal_analysis.md) §8.3)。複数アダプタの登録と選択の扱いは
[vocal_analysis.md](vocal_analysis.md) §8.2 に従う。

| ステージ | 採用ツール | 呼び出し方 | 選定理由 | 代替候補 |
|---|---|---|---|---|
| S0 入力読み込み | **soundfile 優先(mp3も可)+ 自動検出ffmpegにフォールバック**(リポジトリに同梱しない) | 内部ライブラリ/サブプロセス | soundfileで読めない形式のみffmpeg。ffmpegを再配布せずライセンス義務を避ける | —(imageio-ffmpeg 等の同梱配布は不採用) |
| S1 ボーカル抽出 | **Demucs v4 htdemucs_ft**(audio-separator 経由・`shifts=0`) | `audio_separator.separator.Separator`(in-process) | 高品質・MIT・ライブラリ呼び出し可・GPU不要でも動作。生 `demucs.api` は `torchaudio<2.2` 固定で新しい Python 向けビルドが無く不採用 | audio-separator の他モデル(Roformer系等。ライセンス個別確認要) / Spleeter / 分離なし |
| S2 音素・母音認識 | **wav2vec2 音素認識**(transformers + 許諾モデル) | transformers(in-process) | 歌唱頑健性(SSL)・in-process・torchはDemucsと共有・ライセンス清浄。採用確定は S-1 認識ゲート(vocal_analysis.md §9) | Julius 音素認識(phone-loop構成が必要)。Allosaurusは GPL-3.0 で不可 |

S1・S2 は [vocal_analysis.md](vocal_analysis.md) §8.1 のアダプタinterface(Separator / Recognizer)を満たせば
差し替え可能。S0 は固定の内部処理。外部ツールは `vocal_analysis` が内部で呼び、依存は `vocal_analysis` 側に
閉じる。

---

## 5. アダプタ実装メモ(出力の取り込み)

各ツールのネイティブ出力を、[vocal_analysis.md](vocal_analysis.md) §2 の正規化中間形式へ変換する際の要点。

| ツール | ネイティブ出力 | アダプタが取り出すもの |
|---|---|---|
| ffmpeg(自動検出) | 復号したWAV | 復号PCM(チャンネル/サンプルレート保持)のパス。レベル正規化は S0 が施す([vocal_analysis.md](vocal_analysis.md) §3) |
| audio-separator(`Separator.separate`) | API が返す出力ファイルパス(Demucs v4 htdemucs_ft の分離stem) | ボーカルWAVのパス(APIの戻り値を使い、命名を推測しない) |
| wav2vec2 phoneme | フレームごとのCTC音素列(IPA) | 全時間軸被覆のセグメント列(母音/子音/gap+音素ラベル(IPA)+任意の信頼度。IPA→5母音写像は vocal_analysis が提供(RMS不要)、gap の無音/継続判定・無音/閉口の確定は利用先がS3のRMS併用で行い、両唇閉鎖判定は音素から利用先が行う) |
| Julius 音素認識 | アライメント(開始/終了フレーム・音素) | 全時間軸被覆のセグメント列 |

音素→母音への写像規則(IPA→5母音)は `vocal_analysis` が提供し、S-1ゲート採点と利用先の口形イベント確定の
双方が同一規則で使う(写像自体はRMS不要)。音響イベント(連続母音区間・閉鎖・無音)の判断と gap の無音/継続
判定(S3のRMS併用)は利用先(口パク生成系の入口)で行う。外部ツールの非決定性(モデル・スレッド)に注意し、可能な
範囲の決定論を目指す。

---

## 6. ライセンスまとめ

- 本体 MIT([../../LICENSE](../../LICENSE)) / soundfile BSD-3 / libsndfile LGPL-2.1(依存・両立) /
  audio-separator MIT(htdemucs_ft 重みは Demucs v4 由来・MIT)/ transformers Apache-2.0 / torch BSD-3 /
  wav2vec2 モデル `facebook/wav2vec2-lv-60-espeak-cv-ft` Apache-2.0 / 代替候補: Spleeter MIT・
  Julius エンジン 修正BSD / **Allosaurus GPL-3.0=不採用**。
- ffmpeg は同梱・再配布しない(§3)ため、そのビルドのライセンス(LGPL/GPL)による義務は生じない。
- Julius を採用する場合のみ、その音響モデルの個別ライセンスを確認する。

---

## 参考

- Demucs(Python API。元repoは archive、保守 fork が下記): <https://github.com/adefossez/demucs>
- audio-separator: <https://github.com/nomadkaraoke/python-audio-separator>
- Spleeter: <https://github.com/deezer/spleeter>
- Allosaurus(universal phone recognizer): <https://github.com/xinjli/allosaurus>
- wav2vec2 phoneme(transformers): <https://huggingface.co/docs/transformers/en/model_doc/wav2vec2_phoneme>
- Julius(音素認識): <https://github.com/julius-speech/julius>
- imageio-ffmpeg: <https://github.com/imageio/imageio-ffmpeg>
- soundfile: <https://github.com/bastibe/python-soundfile>
- 歌唱ASRの課題(WER比較・ハルシネーション): <https://arxiv.org/abs/2506.15514> ・ <https://arxiv.org/pdf/2403.09298>
