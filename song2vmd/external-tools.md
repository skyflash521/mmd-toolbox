# song2vmd 外部ツール検討資料

`song2vmd` が連携する外部ツールの候補を、パイプラインの各ステージごとに比較し、初期採用と将来の
代替を決めるための資料。仕様本体(連携機構・正規化中間形式)は [song2vmd.md](song2vmd.md) §7 を正とする。
ここで選ぶ初期採用は出発点であり、実データでの品質・導入性の評価で見直す。

## 0. 前提と評価軸

差し替え可能な委譲ステージ([song2vmd.md](song2vmd.md) 4章): S1 ボーカル抽出 / S2 音素・母音認識。
S0 入力読み込みは固定の内部処理(soundfile/ffmpeg)で、差し替えアダプタの対象ではない。強弱解析(RMS)と
モーフ生成は song2vmd 中核(numpy/scipy)で行い、外部ツールに依存しない。

設計上の前提:

- **ユーザーに外部コマンドを叩かせない**。`song2vmd INPUT` の1コマンドで完結し、外部ツールは song2vmd が
  内部で呼ぶ。呼び出しは **ライブラリAPI(Python)を優先**し、無いものだけ内部管理のサブプロセスで呼ぶ。
  依存は song2vmd 側に閉じ、`mmd_toolbox` 本体の必須依存は `numpy/scipy` のまま保つ。
- **歌詞の書き起こし(意味のあるテキスト化)はしない**。口パクに必要なのは「母音の時刻+強弱」であり
  単語ではない。歌唱の自動書き起こしは誤りが多く伝播するため、音声から母音・音素を直接認識する(§2)。
- 本リポジトリはWindowsを主開発環境とする([README.md](../README.md))。ライブラリで導入できること・
  Windows導入容易性を重視する。

評価軸: 日本語適性 / 品質 / 速度・GPU要否 / ライブラリ提供と導入容易性(Windows) / ライセンス /
決定論 / 出力の解析しやすさ / 保守状況。

---

## 1. S1 ボーカル抽出

BGM込み音源からボーカルWAVを得る(S1はステレオ原音入力が高品質。S2向け 16kHz mono 化はS1後)。ボーカル品質はS2の母音認識精度に直結する。ライブラリAPIで
in-process 呼び出しできることを重視する。

| ツール | Python API | 品質 | 速度・要件 | 導入 | ライセンス | 備考 |
|---|---|---|---|---|---|---|
| **Demucs v4 (htdemucs)** | `demucs.api.Separator` | 高(SDR≈9dB) | 中。CPU可/GPUで速い | pip。容易 | MIT | 元repoは2025-01-01 archive(保守終了)。限定bugfix fork: adefossez/demucs |
| **audio-separator** (UVR系) | `Separator` クラス | モデル次第で最高峰 | モデル次第。ONNX | pip。容易。クロスプラットフォーム | MIT | UVRのMDX-Net/VR/Demucs/MDXCを切替 |
| **Spleeter** | あり(TF) | 中(やや古い) | 高速・軽量 | pip(TensorFlow依存) | MIT | 速いが品質は上2者に劣る |

**初期採用: Demucs v4(`demucs.api`)**。理由: 品質・MIT・Python APIでin-process呼び出し可・GPU不要でも動作。
ただし元リポジトリは2025-01-01に archive(保守終了)のため、**限定的なbugfix対応の fork(adefossez/demucs)を版固定で使う**。
再現性のため `shifts=0` 等の非決定要素を固定する。将来の代替に **audio-separator(保守活発・UVRの高品質モデル
を切替)**、軽量・高速の Spleeter。分離不要なボーカル単体入力向けに「分離なし(`never`)」も Separator 実装の
一つとして持つ。

ボーカル抽出ツールの切り替えは [song2vmd.md](song2vmd.md) §7.2 の **Separator 抽象**の背後で行う。Separatorは
出力に「ボーカルWAVのパス」だけを約束し(出力形式は分離器に従う)、内部のライブラリ・モデル・トラック構成は各実装に閉じる。

---

## 2. S2 音素・母音認識(文字起こしを使わない)

### 2.1 なぜ文字起こし(ASR)を使わないか

- **歌唱ASRは誤りが多い**。研究では同一歌詞で歌唱WER≈0.56 / 朗読0.14(約4倍悪化)。伴奏より歌い方の影響が
  大きく、ハルシネーション(無発話区間の捏造)や非語彙発声(ラララ等)に弱い。
- **そもそも単語は不要**。あいうえお口パクに必要なのは「時刻ごとの母音(口形)+強弱」。日本語の5母音は
  音響的に明瞭で、歌唱では母音スペクトルがむしろ安定し、音声から直接認識しやすい。
- **フォースアライメントは正しい歌詞が前提**(与えた文字列に音声を合わせる)ため、歌唱では前提が崩れる。
  一方 **wav2vec2-CTC は文字列なしで音素+時刻を出せる**。Julius も phone-loop(文法なし音素認識)を構成すれば
  文字列なしで音素+時刻を出せる。

→ 文字起こし(S2)+G2P+フォースアライメントを廃し、**文字列なしの音素・母音認識1段**に置換する。

### 2.2 候補

| ツール | 呼び出し | 日本語/汎用 | 時刻精度 | 速度・要件 | ライセンス | 歌唱頑健性 |
|---|---|---|---|---|---|---|
| **wav2vec2 音素認識** | transformers(in-process) | 多言語/日本語 | CTC近似(補正で実用) | CPU可(GPUで速)。torchはDemucsと共有(transformers本体・モデル取得は新規) | transformers=Apache / torch=BSD + 許諾モデル | ◎ 自己教師ありで歌唱に汎化 |
| **Julius 音素認識** | C実行ファイル(内部subprocess) | 日本語(無償音響モデル) | フレーム単位(高) | 軽い。別系統の追加 | エンジン=修正BSD(許諾的) | △ speech-HMMで歌唱は域外 |
| ~~Allosaurus~~ | Python API | 汎用 | 近似 | 軽い | **GPL-3.0 → MIT本体と非互換で不可** | ◎ |

> 補足: CTC系の時刻はトークン単位の粗いオフセットで、音素境界そのものではない。母音区間の境界は認識の
> トークン境界を一次情報とし、近傍にRMSオンセットがあるときだけそれへ寄せて精緻化する。レガートなど同程度の
> 音量が続きオンセットが無い区間はトークン境界をそのまま使う([song2vmd.md](song2vmd.md) 6.3・6.4)。認識器は
> 5母音と子音、未割当(gap)を区別できれば足り(無音・閉鎖の確定は中核がRMS併用で行う)、語彙認識より要件は緩い。

**初期採用: wav2vec2 音素認識(transformers + 許諾モデル)**。決め手:

1. **歌唱頑健性**(本方式の品質の要)で、自己教師あり(SSL)が speech-HMMの Julius に勝る。
2. **純Pythonでin-process**に呼べ、「ユーザーにコマンドを叩かせない/ライブラリ呼び出し」方針に最も合う。
3. **torch を Demucs と共有できる**(Julius は C/モデル/jconf の別系統を追加することになる)。ただし
   transformers 本体とモデル取得(ダウンロード・キャッシュ・メモリ・初回ネットワーク)は新規に必要で、
   モデルの版固定・キャッシュ先・オフライン挙動・必要ディスク/RAM を依存仕様に含める。
4. ライセンスが清浄。transformers=Apache-2.0 / torch=BSD-3 に加え、音素モデル
   `facebook/wav2vec2-lv-60-espeak-cv-ft` は **Apache-2.0(確認済み)**。多言語eSpeak音素を出力し、
   IPA音素→あいうえお母音へ写像する。

弱点の時刻の粗さは、トークン境界を一次情報としRMSオンセットで精緻化して実用化する(6.3・6.4)。**Julius は
「高精度な時刻」の代替**として残す(ただし文字列なし運用には phone-loop の音響モデル・辞書・設定の構成が
必要)。歌唱品質か速度が問題になれば評価する。**Allosaurus は GPL-3.0(LICENSE実物が GNU GPL v3)で
MIT本体と非互換のため不採用**。

なお wav2vec2 の日本語歌唱での母音認識品質は未検証であり、S2はこの一段で製品全体の成否が決まる。
**実装着手前の技術スパイクを必須ゲート**とし、代表となる日本語歌唱サンプルで母音正解率・境界時刻ずれを測り、
あらかじめ定めた受入基準を満たすことを確認してから採用を確定する(§6)。

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

**方針A(自動検出)を採用する(確定)**。利用者環境の ffmpeg を実行時に自動検出(PATH/既知パス)して内部で
呼ぶ。我々は ffmpeg バイナリを一切配らないため、ffmpeg ビルドのライセンス(LGPL/GPL)は我々の義務に
ならない。利用者は ffmpeg を一度だけ導入する(手順は README に記載予定)。`soundfile` で読める形式
(WAV/FLAC/OGG/mp3 等)は soundfile で読み、読めない形式が無ければ ffmpeg 自体が不要なことも多い。

別案として `imageio-ffmpeg` 等でffmpegを同梱配布するpipパッケージに依存すれば導入は楽になるが、同梱
バイナリのビルド・ライセンスがパッケージ依存になるため**採らない**。いずれにせよ利用者に ffmpeg コマンドは
叩かせない。本リポジトリは MIT([../LICENSE](../LICENSE))。

---

## 4. 初期実装で採用するツールと選定理由

**初期実装では各ステージに1ツールだけを採用する**。引数(`--separator` / `--recognizer`)による選択は、
各ステージに2つ目以降のアダプタを追加する段階で有効化する([song2vmd.md](song2vmd.md) §7.4)。

| ステージ | 初期採用 | 呼び出し方 | 選定理由 | 将来の代替 |
|---|---|---|---|---|
| S0 入力読み込み | **soundfile 優先(mp3も可)+ 自動検出ffmpegにフォールバック**(リポジトリに同梱しない) | 内部ライブラリ/サブプロセス | soundfileで読めない形式のみffmpeg。ffmpegを再配布せずライセンス義務を避ける | —(方針B/imageio-ffmpeg は不採用) |
| S1 ボーカル抽出 | **Demucs v4**(adefossez fork・版固定・`shifts=0`) | `demucs.api`(in-process) | 高品質・MIT・ライブラリ呼び出し可・GPU不要でも動作 | audio-separator(保守活発) / Spleeter / 分離なし |
| S2 音素・母音認識 | **wav2vec2 音素認識**(transformers + 許諾モデル) | transformers(in-process) | 歌唱頑健性(SSL)・in-process・torchはDemucsと共有・ライセンス清浄。実装前に実現可能性ゲート(§2.2) | Julius 音素認識(phone-loop構成が必要)。Allosaurusは GPL-3.0 で不可 |

S1・S2 は [song2vmd.md](song2vmd.md) §7.2 のアダプタinterface(Separator / Recognizer)を満たせば差し替え可能。
S0 は固定の内部処理。外部ツールは song2vmd が内部で呼び、依存は song2vmd 側に閉じる。

---

## 5. アダプタ実装メモ(出力の取り込み)

各ツールのネイティブ出力を、[song2vmd.md](song2vmd.md) §7.3 の正規化中間形式へ変換する際の要点。

| ツール | ネイティブ出力 | アダプタが取り出すもの |
|---|---|---|
| ffmpeg(自動検出) | 復号したWAV | 復号PCM(チャンネル/サンプルレート保持)のパス。レベル正規化は S0 が施す([song2vmd.md](song2vmd.md) §6.1) |
| Demucs (`demucs.api`) | 分離stem(配列/ファイル) | ボーカルWAVのパス(分離器出力) |
| audio-separator | API が返す出力ファイルパス | ボーカルWAVのパス(APIの戻り値を使い、命名を推測しない) |
| wav2vec2 phoneme | フレームごとのCTC音素列(IPA) | 全時間軸被覆のセグメント列(母音/子音/gap+音素ラベル(IPA)+任意の信頼度。gap・無音・閉鎖の確定とあいうえお写像は中核がS3のRMS併用で行う) |
| Julius 音素認識 | アライメント(開始/終了フレーム・音素) | 全時間軸被覆のセグメント列 |

音素→母音への写像と音響イベント(連続母音区間・閉鎖・無音)の判断、および gap の無音/継続判定(S3のRMS
併用)は中核側(日本語固有処理)で行い、アダプタは「音声→母音/子音/gap のセグメント列」だけを返す(RMS不要)。
外部ツールの非決定性(モデル・スレッド)に注意し、可能な範囲の決定論を目指す。

---

## 6. 確認事項・次アクション

- **S2 実現可能性ゲート(最優先・品質)**: 採用ツール・モデルは確定済み(`facebook/wav2vec2-lv-60-espeak-cv-ft`、
  Apache-2.0)。残るは品質検証で、代表となる日本語歌唱サンプルで母音正解率・境界時刻ずれを実測し、先に定めた
  受入基準を満たすことをゲートにする。S2が成立しないと製品全体が成立しないため、満たさなければ
  Julius(phone-loop構成)等を評価する。受入基準・代表データ・指標は実装で確定する。
- **ライセンス(確定)**: 本体 MIT / soundfile BSD-3 / libsndfile LGPL-2.1(依存・両立) / Demucs(adefossez fork・
  htdemucs 重み)MIT / transformers Apache-2.0 / torch BSD-3 / wav2vec2 モデル(上記)Apache-2.0 / 代替:
  audio-separator MIT・Spleeter MIT・Julius エンジン BSD-3 / **Allosaurus GPL-3.0=不採用**。ffmpeg は方針A
  (非再配布)で義務なし。Julius を採用する場合のみ、その音響モデルの個別ライセンスを確認する。
- ffmpeg: 方針A(自動検出・非再配布)を採用(確定)。soundfile(mp3 含む)で読めない形式のみ ffmpeg。
- 決定論の確認: 同一入力・同一モデルでの出力安定性(Demucs `shifts=0`、スレッド/シード固定)。

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
