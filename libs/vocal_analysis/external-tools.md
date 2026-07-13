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
- **書き起こしテキストを製品の入出力にしない**。後段に必要なのは「母音・音素の時刻」であり単語ではない。
  書き起こしテキストは共有出力に含めず、歌詞テキストの入力も要求しない([vocal_analysis.md](vocal_analysis.md) §5)。
  ただし認識構成が**内部で**内容認識(文字起こし)を音素列の情報源として使うことは、この方針に反しない(§2)。
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

## 2. S2 音素・母音認識

### 2.1 認識方式の前提(実歌唱データでの測定に基づく)

- **単段の自由音素認識(CTC)は歌唱で不成立**。話し言葉では音素を出力する CTC 音素認識
  (wav2vec2-espeak)が、歌唱では blank(未割当)が支配的になり(blank 事後確率が平均9割前後)、母音を
  ほぼ出力しない(参照ラベル付き歌唱データでの実測。母音正解率が数%に落ちる)。デコード方式の変更
  (argmax・カテゴリ確率集約)でも覆らず、モデルの限界である。
- **内容認識(文字起こし)は書き起こされた区間では歌唱の内容をおおむね再現するが、長尺の単一呼び出しでは
  幻覚が出る**。大規模 ASR(Whisper 系)は、書き起こしが機能している区間では歌唱の内容をおおむね正しく
  文字起こしできる(軽微な単語誤りは残る。精度の具体的な数値評価はS-1測定=9章に委ねる)。単語レベルの
  タイムスタンプは、直接の時刻源として単体で使うには粗いが、±0.75秒程度の余白を持つ強制アライメントの
  制約源としては有効であることを実測で確認済み(vocal_analysis.md §5.2 手順3・手順7)。また、
  数分規模の入力を単一呼び出しで渡すと、反復幻覚(同一文の繰り返し)や
  無音・低エネルギー区間での定型句幻覚(動画配信由来の学習データに起因する「ご視聴ありがとうございました」
  等)を起こす(参照ラベル付きフル曲データでの実測)。採用構成はこれに対処するため無音検出で区間分割し、
  区間ごとに独立して Whisper を呼ぶ([vocal_analysis.md](vocal_analysis.md) §5.2)。
- **強制アライメントは歌唱でも時刻をある程度正確に復元できるが、素朴な一括適用には落とし穴がある**。
  既知の音素列に CTC トレリスの経路を拘束する強制アライメント(Viterbi)なら、blank が支配的な歌唱でも、
  blank 支配下の自由デコードよりはるかに良く音素の開始時刻を復元できる。ただし音素列全体をボーカルWAV
  全体へ一括で対応付けると、blank にマップされる状態(列の先頭・末尾の無音等)がほぼゼロコストで留まり
  続けられるため、実際の音素列全体をごく短い窓へ押し込んでしまう崩壊が起こりうる(参照ラベル付きデータ
  での実測)。採用構成はこれに対処するため、単語タイムスタンプに基づく単語窓制約を主とし、単語
  タイムスタンプが取得できない場合のフォールバックとして位置バンド制限も設ける
  ([vocal_analysis.md](vocal_analysis.md) §5.2)。「正しい歌詞が前提」という強制アライメントの弱点は、
  歌詞を音声自身から内容認識で得ることで避ける。CTC の出力は音素位置のスパイクであり持続長を持たないため、
  母音の終端(閉じ側)は近似になる(閉じ側の確定は利用先の入口処理の責務。
  [vocal_analysis.md](vocal_analysis.md) §9・§10)。時刻精度の具体的な数値は測定条件により変わるため
  ここには固定せず、vocal_analysis.md §9 の S-1 測定で確認する。

→ この3点から、S2 の有望構成は**複合構成**とする: 内容認識(Whisper 系)で歌唱内容のテキストを得て、
G2P(pyopenjtalk 系)で音素列へ変換し、CTC 音素モデルのロジット上の強制アライメントで時刻を確定する。
書き起こしは共有出力に出さず、音素列の情報源としてアダプタ内部でだけ使う(§0 の方針と整合)。

### 2.2 候補

内容認識には、どのモデルでも共通のかな限定プロンプト(vocal_analysis.md §5.2・§8.3)を渡す(モデルに
よる分岐は無い)。

| 構成 | 呼び出し | 時刻精度 | 歌唱での成立 | 語彙置換のリスク | ライセンス |
|---|---|---|---|---|---|
| **`openai/whisper-medium` + G2P + CTC強制アライメント(既定値)** | transformers + pyopenjtalk-plus(いずれも in-process) | 強制アライメントで復元 | ○(歌唱データで検証済み) | 中(かな化成功区間は低、失敗区間は通常のWhisper単体と同等) | Whisper モデル Apache-2.0 / pyopenjtalk-plus MIT |
| **`kana-whisper` + G2P + CTC強制アライメント(候補値)** | transformers + pyopenjtalk-plus(いずれも in-process) | 強制アライメントで復元(具体的な精度はvocal_analysis.md §9のS-1測定で確認) | 未評価(歌唱データでの学習・評価実績が無い) | 低(意味の通る別語への丸ごと置換は起きにくい) | kana-whisper MIT / pyopenjtalk-plus MIT(内包の OpenJTalk 系は修正BSD) |
| wav2vec2 自由音素認識(単段。音素/かな出力とも) | transformers(in-process) | CTC近似 | ×(実測で不成立: blank支配でほぼ何も出力しない) | 低(意味補正なし) | transformers=Apache / torch=BSD + 許諾モデル |
| Julius 音素認識(phone-loop) | C実行ファイル(内部subprocess) | フレーム単位(高) | 未評価(speech-HMM で歌唱は域外の懸念。phone-loop の構成も必要) | 未評価 | エンジン=修正BSD(音響モデルは個別確認) |
| ~~Allosaurus~~ | Python API | 近似 | 未評価 | 未評価 | **GPL-3.0 → MIT本体と非互換で不可** |

> 補足: 強制アライメントの時刻は音素の開始側が終端より相対的に正確で、終端(閉じ側)は近似となる
> (歌唱では開始側にもフレーズ内のずれがある。vocal_analysis.md §5.2 の既知の限界)。母音区間の境界の精緻化
> (近傍にRMSオンセットがあるときそれへ寄せる)と閉じ側の確定は利用先(口パク生成系の入口)が行う
> ([vocal_analysis.md](vocal_analysis.md) §5・§10)。認識器は5母音と子音、未割当(gap)を区別できれば足り(無音/閉口の
> 確定は利用先がRMS併用で行い、両唇閉鎖は音素から判定)、語彙認識より要件は緩い。

**採用: 内容認識+G2P+CTC強制アライメントの複合構成(id・モデル・revision の固定は
vocal_analysis.md §5.2・§8.3 が正本)。内容認識は既定値 `openai/whisper-medium`、候補値
`kana-whisper`(いずれもかな限定プロンプトを渡す)**。決め手:

1. **既定値は歌唱データで検証済み・高速**。`openai/whisper-medium`+かな限定プロンプトは既存のS-1
   測定で歌唱データにおける成立が確認済みで、`kana-whisper` より大幅に高速(具体的な倍率は実行環境に
   依存するため恒久値として固定しない)。かな限定プロンプトの効果が及ばず漢字混じりの書き起こしに
   留まった区間は、`pyopenjtalk-plus` の辞書・形態素解析が読みを決めるため同字異音の読み違いが
   残りうる(vocal_analysis.md §5.2)。
2. **口パク生成系が必要とするのは母音の種類(母音正解率)であり、意味の通る文か否かではない**。内容認識に
   強い言語モデル的補正を持つ構成(通常のWhisper書き起こし)は、聞き取りに自信が持てない区間で実際の
   発声と無関係な別の語へ丸ごと置き換えることがあり、この場合は母音自体が変わり口形が破綻する
   (参照ラベル付き歌唱データとの実測比較で確認済み)。かなを直接出力し言語モデル的補正を持たない
   `kana-whisper` はこの種の語彙置換を起こしにくく、候補値として提供する。ただし歌唱データでの
   学習・評価実績が無い。
3. **単段の自由CTC認識(音素/かな出力を問わず)は歌唱で不成立**。blank(未割当)が支配的になりほぼ
   何も出力しない(参照ラベル付き歌唱データでの実測。wav2vec2ベースの複数の出力語彙で確認済み)。
   `kana-whisper` は自由CTCではなく Whisper と同じ系列変換(自己回帰デコーダ)であるため、この不成立を
   回避できる。
4. **G2P(`pyopenjtalk-plus`)はどちらの内容認識モデルの出力に対しても同一の呼び出しで足りる**。入力が
   かなであれば、漢字の読みに起因する曖昧性(同字異音の読み違い)は構造的に生じない。ただし助詞の
   読みや長音の解釈などかなでも文脈依存のケースはわずかに残る(vocal_analysis.md §5.2)。
5. **すべて純Pythonでin-process**に呼べ、「利用者にコマンドを叩かせない/ライブラリ呼び出し」方針に合う。
   torch・transformers を既存の S1(Demucs)・アライメント用 CTC モデルと共有できる。内容認識モデルの
   実行デバイスは環境依存で自動選択し(GPUが利用可能ならGPUを使う)、CPU実行時もスレッド数を制限しない
   (アライメント用音素モデルはCPU・単一スレッド固定のまま。決定論を含む詳細は
   [vocal_analysis.md](vocal_analysis.md) §5.1・§5.2 が正本)。この自動選択は実行時の分岐であり、
   導入する torch 自体がCUDA対応ビルドかどうかは pip の既定インストールでは選べない
   (CPU専用ビルドがPyPI本体の既定で、CUDA対応ビルドは別indexでのみ配布されるため)。GPU利用は
   利用者が任意でCUDA対応ビルドを追加導入した場合の効果に留まり、既定インストール(CPU専用ビルド)
   でもGPU不要という前提どおり確実に動く。S1(audio-separator)も torch の
   `torch.cuda.is_available()` を自ら見てGPUを自動選択する(ライブラリ側の既存挙動)ため、
   同じ前提を共有する。
6. ライセンスが清浄。kana-whisper = MIT、Whisper モデル(Hugging Face 配布)= Apache-2.0、
   pyopenjtalk-plus = MIT(内包の OpenJTalk 系コンポーネントは修正BSD)、アライメント用音素モデル
   `facebook/wav2vec2-lv-60-espeak-cv-ft` = Apache-2.0。

実装(アダプタ)は [vocal_analysis.md](vocal_analysis.md) §5.2・§8.3 の確定仕様に従う。実装後の
最終的な品質確認は **S-1 認識測定**(構成間の相対比較・破綻検出)と利用先の実装時調整・MMD 上の視聴確認に
従う([vocal_analysis.md](vocal_analysis.md) §9)。**Julius は代替**として残す(歌唱品質か速度が問題に
なれば評価する)。**Allosaurus は GPL-3.0(LICENSE実物が GNU GPL v3)で MIT 本体と非互換のため不採用**。

### 2.3 SOFA(強制アライメント段の選択制代替候補)

SOFA(Singing-Oriented Forced Aligner。<https://github.com/qiuqiao/SOFA>。MIT)は、既知の音素記号列を
音声へ時刻合わせする強制アライナー本体で、自由音素認識(ASR)機能は持たない。2.1の複合構成のうち
強制アライメント段(手順5以降相当)だけを置き換える選択肢として、`vocal_analysis.md` §5.3・§8.3の
`sofa-forcedalign`アダプタとして選択可能である(既定は2.1の複合構成`wav2vec2-ctc-forcedalign`のまま)。

- **単独比較での品質**: 参照ラベル付き歌唱コーパスでの単独比較(検証専用に入手したチェックポイントを
  使用)では、既定構成(wav2vec2 CTC強制アライメント)より母音一致率・境界時刻精度とも上回った。
- **フルパイプライン比較での乖離**: 内容認識→G2P→強制アライメント→VMD生成のフルパイプラインで
  実曲を比較すると、母音一致率等の数値指標では両者が同等以上に見える条件でも、実際にMMD上で
  目視・耳確認すると既定構成の方が自然に見える乖離が生じた。この乖離が
  `vocal_analysis.md` §8.3で両アダプタを選択可能なまま維持し既定を変更しない理由の根拠である。
- **チェックポイントのライセンス(既定値を持たない理由)**: 入手可能な日本語SOFAチェックポイント
  (Greenleaf2001/SOFA_Models・colstone/SOFA_Models)はいずれも商用利用が制限されている
  (前者は商用利用不可を明記、後者は訓練データに商用利用に個別相談を要するコーパスを含む)。この2件を
  含め、チェックポイント(モデル重み)を本リポジトリへ同梱すると重みデータの再配布に当たり上記の
  ライセンス条件に反するため同梱しない。同梱しない場合でも、商用利用不可・個別相談要のチェックポイントを
  既定値にすると、素の状態でツールを動かした利用者(商用利用を含む)が制限に気づかないまま使ってしまい
  意図せずライセンス違反を犯しうるため、`vocal_analysis.md` §8.3のとおりこの2件のいずれも既定値には
  しない(利用者保護のための方針判断)。
- **依存非互換とサブプロセス方式の採用理由**: SOFAのPython依存(固定版のnumpy・librosa等)は本リポジトリの
  実行環境(numpy 2.x系)と非互換のため、同一環境に同居させず、利用者が用意する専用venvをサブプロセス
  として呼ぶ方式を採る。SOFA本体のコードも同様の依存非互換によりリポジトリへ同梱できない。
- **既知のトレードオフ**: SOFAの`force`モードは、既定構成の単語窓制約(内容認識が返す単語ごとの推定
  時刻範囲に余白を加えた到達可能フレーム範囲で状態遷移を制約する仕組み)に相当する外部からの
  時間的制約注入を持たないため、単語単位で音声チャンクを切り出してアライメントする
  (`vocal_analysis.md` §5.3)。この単語境界はマージン無しの物理的な音声チャンク境界になり、
  既定構成の余白(マージン)付き到達可能窓より単語境界での協調調音・語尾の伸びを吸収する柔軟性が
  低い。また15ms未満の極端に短い音素セグメントを既定構成より多く出力する傾向がある(利用先の
  後処理である程度吸収されるが完全に無害とは確認できていない)。

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

S0・S1のアダプタは1つ(採用ツールの正本は [vocal_analysis.md](vocal_analysis.md) §8.3)。S2は
強制アライメント段のアダプタを`forced_aligner`引数で選択できる(既定アダプタ`wav2vec2-ctc-forcedalign`・
選択制アダプタ`sofa-forcedalign`)。既定アダプタを選んだ場合はさらに、内容認識モデルを
`content_recognizer_model` 引数(既定値・候補値・任意指定。vocal_analysis.md §5.2)で選べる。
アダプタの登録と選択の扱いは [vocal_analysis.md](vocal_analysis.md) §8.2 に従う。

| ステージ | 採用ツール | 呼び出し方 | 選定理由 | 代替候補 |
|---|---|---|---|---|
| S0 入力読み込み | **soundfile 優先(mp3も可)+ 自動検出ffmpegにフォールバック**(リポジトリに同梱しない) | 内部ライブラリ/サブプロセス | soundfileで読めない形式のみffmpeg。ffmpegを再配布せずライセンス義務を避ける | —(imageio-ffmpeg 等の同梱配布は不採用) |
| S1 ボーカル抽出 | **Demucs v4 htdemucs_ft**(audio-separator 経由・`shifts=0`) | `audio_separator.separator.Separator`(in-process) | 高品質・MIT・ライブラリ呼び出し可・GPU不要でも動作。生 `demucs.api` は `torchaudio<2.2` 固定で新しい Python 向けビルドが無く不採用 | audio-separator の他モデル(Roformer系等。ライセンス個別確認要) / Spleeter / 分離なし |
| S2 音素・母音認識(既定アダプタ) | **複合構成(内容認識 + G2P + wav2vec2 CTC 強制アライメント)を採用**。内容認識は既定値 `openai/whisper-medium`、候補値 `kana-whisper`(いずれもかな限定プロンプトを渡す。単段の自由CTC認識は歌唱で不成立と実測済み。§2。id・モデル・revision の固定は vocal_analysis.md §5.2・§8.3 が正本) | transformers + pyopenjtalk-plus(in-process) | 既定値は歌唱で検証済み・高速。候補値は語彙置換による母音破綻が起きにくい。いずれもin-process・torch/transformersは既存と共有・ライセンス清浄 | Julius 音素認識(phone-loop構成が必要)。Allosaurusは GPL-3.0 で不可 |
| S2 音素・母音認識(選択制アダプタ) | **SOFA(単語単位アライメント)。利用者提供の専用venv・チェックポイントが必須(既定値なし。§2.3)** | 利用者提供の専用venvをサブプロセス呼び出し | 単独指標だけではフルパイプラインの最終品質の優劣を判定できないため、既定アダプタと並ぶ選択肢として提供する(§2.3) | ―(SOFA自体が既定アダプタへの代替候補) |

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
| 複合構成(内容認識 + G2P + CTC強制アライメント。既定値=whisper-medium・候補値=kana-whisper) | 内容認識: テキスト(既定値はかな化されない場合あり、候補値は常にかな) / G2P: 音素列 / アライメント: 音素ごとの開始位置(CTCスパイク) | 全時間軸被覆のセグメント列(母音/子音/gap+音素ラベル+任意の信頼度)。テキストと音素列はアダプタ内部にとどめ、共有出力に含めない。母音の終端(閉じ側)はスパイク位置からの近似で、確定は利用先(S3のRMS併用) |
| SOFA(単語単位アライメント。選択制) | HTK形式の音素ごとの開始/終了時刻(100ナノ秒単位)。AP(呼吸音)・SP(無音)ラベルを含む | 全時間軸被覆のセグメント列(母音/子音/gap+音素ラベル(IPA)+任意の信頼度)。100ナノ秒単位の時刻を秒へ変換し、Segment契約(隙間なく連続・非重複で全時間軸を被覆)を検証してから正規化する。AP・SPはgapへ写像する(vocal_analysis.md §5.3) |
| wav2vec2 phoneme(単段自由認識) | フレームごとのCTC音素列(IPA) | 全時間軸被覆のセグメント列(母音/子音/gap+音素ラベル(IPA)+任意の信頼度。IPA→5母音写像は vocal_analysis が提供(RMS不要)、gap の無音/継続判定・無音/閉口の確定は利用先がS3のRMS併用で行い、両唇閉鎖判定は音素から利用先が行う) |
| Julius 音素認識 | アライメント(開始/終了フレーム・音素) | 全時間軸被覆のセグメント列 |

音素→母音への写像規則(IPA→5母音)は `vocal_analysis` が提供し、S-1認識測定の採点と利用先の口形イベント確定の
双方が同一規則で使う(写像自体はRMS不要)。音響イベント(連続母音区間・閉鎖・無音)の判断と gap の無音/継続
判定(S3のRMS併用)は利用先(口パク生成系の入口)で行う。外部ツールの非決定性(モデル・スレッド)に注意し、可能な
範囲の決定論を目指す。

---

## 6. ライセンスまとめ

- 本体 MIT([../../LICENSE](../../LICENSE)) / soundfile BSD-3 / libsndfile LGPL-2.1(依存・両立) /
  audio-separator MIT(htdemucs_ft 重みは Demucs v4 由来・MIT)/ transformers Apache-2.0 / torch BSD-3 /
  wav2vec2 モデル `facebook/wav2vec2-lv-60-espeak-cv-ft` Apache-2.0 / kana-whisper モデル(Hugging Face
  配布)MIT / Whisper モデル(Hugging Face 配布)Apache-2.0 / pyopenjtalk-plus MIT(内包の
  OpenJTalk・hts_engine は修正BSD系) / 代替候補: Spleeter MIT・Julius エンジン 修正BSD /
  **Allosaurus GPL-3.0=不採用**。
- SOFA本体(<https://github.com/qiuqiao/SOFA>)MIT。SOFAチェックポイント(モデル重み)は本リポジトリへ
  同梱しない(§2.3)。同梱しないチェックポイント自体のライセンス条件の遵守は、チェックポイントを
  指定する利用者自身の選択・責任の範囲。
- ffmpeg は同梱・再配布しない(§3)ため、そのビルドのライセンス(LGPL/GPL)による義務は生じない。
- Julius を採用する場合のみ、その音響モデルの個別ライセンスを確認する。
- 英語未知語カタカナ化フォールバック([vocal_analysis.md](vocal_analysis.md) §5.2手順4)関連:
  arpakana MIT(既定の変換方式) / nltk
  Apache-2.0(CMUdict取得用)/ CMUdict(`nltk.corpus.cmudict`が配布するコーパス本体)修正BSD /
  tinyllama-katakana-converter モデル(Hugging Face 配布 `pyon0024/tinyllama-katakana-converter`。
  選択式の変換方式)Apache-2.0。

---

## 参考

- Demucs(Python API。元repoは archive、保守 fork が下記): <https://github.com/adefossez/demucs>
- audio-separator: <https://github.com/nomadkaraoke/python-audio-separator>
- Spleeter: <https://github.com/deezer/spleeter>
- Allosaurus(universal phone recognizer): <https://github.com/xinjli/allosaurus>
- wav2vec2 phoneme(transformers): <https://huggingface.co/docs/transformers/en/model_doc/wav2vec2_phoneme>
- Whisper(transformers): <https://huggingface.co/docs/transformers/en/model_doc/whisper>
- kana-whisper: <https://huggingface.co/sbintuitions/kana-whisper>
- pyopenjtalk-plus(G2P): <https://github.com/tsukumijima/pyopenjtalk-plus>
- Julius(音素認識): <https://github.com/julius-speech/julius>
- SOFA(Singing-Oriented Forced Aligner): <https://github.com/qiuqiao/SOFA>
- imageio-ffmpeg: <https://github.com/imageio/imageio-ffmpeg>
- soundfile: <https://github.com/bastibe/python-soundfile>
- 歌唱ASRの課題(WER比較・ハルシネーション): <https://arxiv.org/abs/2506.15514> ・ <https://arxiv.org/pdf/2403.09298>
