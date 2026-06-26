# vpr_io 要求仕様書

VOCALOID プロジェクトファイル(vpr)の読み書きを担う**形式I/Oモジュール**の要求仕様。
vpr はMMD形式ではないため、MMD形式層(`mmd_toolbox`)とは別のモジュールとして置く。

---

## 1. 位置づけ

### 1.1 目的

VOCALOID のプロジェクトファイル(vpr)を読み書きし、利用先が必要とする音楽情報を正規化したデータモデル
(2章)として提供する。複数の利用先で再利用するため共有モジュールとし、各CLIに埋め込まない(CLIに埋めると
CLI間依存になる)。

### 1.2 設計境界

CLAUDE.md の設計境界に従う。**MMDでない形式(VOCALOID vpr)の入出力を担う専用の形式I/Oモジュール**。

- vpr の解析・直列化(形式の事実)は `vpr_io` に置く。
- **形式の事実のみを扱う**(`mmd_toolbox` と同じ原則): ツール固有の判断は持ち込まない。vpr から口形イベント列・
  開き量を作る、ピッチ推定、音符分割・歌詞/音素対応付けなどの生成・判断は各CLI(`vpr2vmd`/`song2vpr`)や
  共有ドメイン(`lipsync`)の責務。
- VMD形式の入出力は `mmd_toolbox`、vpr形式の入出力は `vpr_io` が担う(形式ごとに別モジュール)。

### 1.3 利用先

| 利用先 | 用途 | 方向 |
|---|---|---|
| `vpr2vmd` | 音符の時刻・歌詞/音素(→口形イベント)・強弱(→開き量)・休符を読む | 読み |
| S-1認識ゲートのラベル生成 | 音符の時刻・音素(→母音/子音/休符の自動ラベル)を読む | 読み |
| `song2vpr` | 音符(時刻・ピッチ・歌詞/音素・強弱)・テンポを vpr へ書き出す | 書き |

読みは `vpr2vmd`・S-1ゲートで先に必要になり、書きは `song2vpr` 着手時に必要になる(実装は読みを先行する)。

---

## 2. データモデル(vpr_io が公開する抽象)

vpr の音楽情報のうち、利用先(1.3)が必要とするものを、特定のCLIに依らない正規化データモデルとして公開する。

- **テンポ・拍子**: 時刻と vpr の時間表現(tick/拍)の相互変換に必要な情報。
- **トラック/パート構造**: 歌唱トラックとその中のパート(歌唱区間)。
- **音符(ノート)**: 開始時刻・長さ(タイミング)、ピッチ(音高)、歌詞、音素(発音)、強弱。正規化公開モデルの
  強弱は**ベロシティ(0〜127 の生値)**とする(下記「確定事項」)。より細かなダイナミクス曲線は公開モデルに
  含めず、保持する場合はロスレス未解釈データ側(§3・§5)に置く。
- **休符**: 音符間の空き(無音区間)として観測できる情報。

利用先が使う部分集合は 1.3 のとおり。`vpr2vmd` とS-1ゲートは時刻・音素・(vpr2vmd は)強弱を使い、`song2vpr`
は音符全体とテンポを書き出す。データモデルは利用先の和集合を表現できれば足り、利用しないフィールドの
読み出しは各利用先が選ぶ。

**確定事項**(机上で確定する。具体的な型・公開関数シグネチャは §2.1。
形式レイアウト確定でモデルが変わる場合は本書を先に更新する。§5・§6):

- **時刻の単位・座標系**: vpr ネイティブの **tick(整数)** で保持し、音符の時刻は**プロジェクト絶対 tick**で
  正規化する(パート相対で格納する vpr は読み込み時に絶対化する)。秒/フレーム/拍への変換は呼び出し側が行う
  (§6)。tick⇔秒変換に必要なテンポマップ・分解能(tick/四分音符)・拍子も公開する。
- **音素の粒度**: 音素は **音符単位** とし、各音符が音素列(発音記号の並び)を持つ。音符内の音素別タイミングは
  初期スコープ外(必要になれば拡張)。
- **強弱の表現**: 各音符の **ベロシティ(0〜127 の生値)** を持つ。より細かなダイナミクス曲線の保持は
  ロスレス範囲(§3・§5、形式レイアウト確定後)に委ねる。
- **休符**: 明示の型を持たせず、同一トラック内の**発音区間(各音符の [開始, 開始+長さ))の和集合の補集合**として
  観測する(隣接差分でなく和集合の補集合とすることで、万一区間が重なっても偽の休符を作らない)。先頭音符より
  前・解析範囲末尾までの無音を含み、解析範囲は呼び出し側が定める。歌唱トラックは単音想定(発音区間の重なりを
  想定しない)で、重なる入力は `read` が `VprWarning` で報告する。
- **公開関数**: 読み(§3)とデータモデル、書き(§4)の2方向。読みはデータモデルを返し、書きはデータモデルを
  vpr へ直列化する。

### 2.1 公開する具体型と関数

型は Python の dataclass、公開関数はモジュール関数とし、`mmd_toolbox.vmd`(`types.py` / `io.py`)の作法に倣う。
時刻は vpr ネイティブの **tick(整数)** で保持し、秒/フレーム/拍への変換は持たない(§6)。

- `VprProject`: `resolution: int`(tick/四分音符)、`tempos: list[TempoEvent]`、
  `time_signatures: list[TimeSignature]`、`tracks: list[Track]`、
  `raw_sequence: dict | None`(未解釈データのロスレス保持。§3。`read` は `sequence.json` 全体を保持し、
  手組みの `VprProject` では `None`)。
- `TempoEvent`: `tick: int`、`bpm: float`。
- `TimeSignature`: `tick: int`、`numerator: int`、`denominator: int`。
- `Track`: `name: str`、`parts: list[Part]`。
- `Part`: `name: str`、`start_tick: int`(プロジェクト絶対 tick。パートの開始位置)、
  `notes: list[Note]`(`start_tick` の昇順)。
- `Note`: `start_tick: int`、`duration_tick: int`、`pitch: int`(MIDI ノート番号)、`lyric: str`(表示歌詞)、
  `velocity: int`(0〜127)、`phonemes: list[str]`(音符内の音素列。空可、既定は空リスト)。
  `start_tick` は**プロジェクト絶対 tick**、`duration_tick` は **tick 長**で持つ(vpr がパート相対で格納する場合、
  read 時に `Part` 開始位置を `start_tick` へ加算して絶対化する)。
- **休符**: 専用型を持たせない。同一 `Track` 内の発音区間 `[start_tick, start_tick+duration_tick)` の和集合の
  補集合を休符とする(上記「確定事項」)。
- **警告/エラー**: `VprWarning`(`code: str`、`message: str`。続行可能事象の構造化報告)、
  `VprFormatError`(構造異常の例外)。`mmd_toolbox.vmd.types` の `VmdWarning`/`VmdFormatError` に倣う。
  - `VprWarning` はロケータとして `track_index: int | None`、`part_index: int | None`、
    `note_index: int | None`、`related_note_index: int | None`、`tick: int | None` を持つ(既定 `None`)。
    `track_index`/`part_index`/`note_index` は `VprProject.tracks[]`/`Track.parts[]`/`Part.notes[]` の添字、
    `related_note_index` は2音符の関係(重なり等)を報告する場合の相手側 `Part.notes[]` の添字、`tick` は対象位置の
    プロジェクト絶対 tick。`section`/`frame` のような VMD 固有のロケータは持たない(vpr に対応概念が無いため)。
    `code` は英小文字の `snake_case`(例: `overlapping_notes`)。
  - `VprFormatError` は原因特定のため `message: str`、`path: str | None`、`key: str | None`、
    `value: object | None` を持つ。`path` は ZIP エントリ名または JSON パス(例 `tracks[2].parts[0].notes[4].pos`)、
    `key` は欠落・型不一致の対象キー、`value` は問題になった実値。詳細な範囲は §3。
- **公開関数**(`mmd_toolbox.vmd.io` に倣う):
  - `read(src: str | Path | bytes) -> tuple[VprProject, list[VprWarning]]`(読み)。
  - `write(project: VprProject) -> bytes` / `write_file(project: VprProject, path: str | Path) -> None`
    (書き、`song2vpr` 着手時)。`write_file` は原子置換(同ディレクトリ一時ファイルへ書いて fsync し
    `os.replace`)で、書き込み途中の中断・ディスクフルでも既存の出力先を破損させない
    (`mmd_toolbox.vmd.io.write_file` に倣う)。
- 未解釈データ(ロスレス保持)は `VprProject.raw_sequence` に `sequence.json` 全体を保持する範囲とする
  (保持範囲と往復保証の限界は §3・§5)。オブジェクト単位の未マップキー保持(`extras` 等)は必要が確認された
  時点で `song2vpr`(write)で検討する。

---

## 3. 読み込み(read)

- vpr を読み、データモデル(2章)を返す。
- **ロスレス原則**(`mmd_toolbox` に倣う): 解釈しないデータも可能な範囲で保持し、無加工なら書き戻せること
  (読み→書きの再現性)を目標とする。保持の範囲は形式レイアウトの確定(5章)に依存する。
- **CLI非依存・出力なし**: 形式の異常は構造化エラーで返し、続行可能な事象は構造化された警告で返す
  (表示の判断は呼び出し側)。`print`/`logging` へ直接出力しない。

`read` は形式仕様([docs/specs/vpr/VPR_file_format.md](../docs/specs/vpr/VPR_file_format.md))の `sequence.json` を
次のとおりデータモデル(§2.1)へ写像する(フィールドのレイアウトは形式仕様を正とする):

- `VprProject.resolution` = 480(形式仕様の固定分解能)。
- `VprProject.tempos` ← `masterTrack.tempo.events`。`TempoEvent(tick=pos, bpm=value/100)`。
- `VprProject.time_signatures` ← `masterTrack.timeSig.events`。小節番号 `bar` を tick へ変換
  (分解能と先行する拍子から各小節の tick 長を積算)して `TimeSignature(tick, numerator=numer, denominator=denom)`。
- `VprProject.tracks` ← `tracks` のうち歌唱トラック(`type` = 2)。`Track(name, parts)`。オーディオトラック
  (`type` = 1)は音符を持たず公開データモデルに現れない(その保持は §3.3)。
- `Part(name, start_tick=part.pos, notes)`。`notes` は `start_tick` 昇順。
- `Note(start_tick=part.pos + note.pos, duration_tick=duration, pitch=number, lyric, velocity, phonemes=phoneme.split())`。
- 休符は専用型を持たず、§2.1 のとおり発音区間の和集合の補集合として導出する。

### 3.1 構造異常(`VprFormatError`)

`read` は、vpr コンテナまたは `Project/sequence.json` の必須構造を読み取れず公開データモデル(2章)へ写像できない
入力を `VprFormatError` で報告する。続行可能な事象(警告)と異なり、正しい `VprProject` を構築できないため例外で返す。

- **`VprFormatError` とする入力**:
  - 入力が ZIP アーカイブでない、または ZIP として破損している。
  - ZIP 内に `Project/sequence.json` が存在しない。
  - `Project/sequence.json` を UTF-8 JSON として解析できない。
  - トップレベル必須キー `masterTrack` または `tracks` が存在しない。
  - `masterTrack.tempo.events[]` / `masterTrack.timeSig.events[]` から `TempoEvent` / `TimeSignature` を構築するための
    必須キーまたは型が不正である。
  - 歌唱トラック・パート・音符の公開モデル対象フィールドが欠落・型不正である。音符は `pos`・`duration`・`number`・
    `lyric`・`phoneme`・`velocity` の6フィールドを必須とする(形式仕様 VPR_file_format.md)。
- **`VprFormatError` としない入力(許容)**:
  - 公開データモデル対象外の未知キー・未解釈キー(ロスレス保持側 §3.3 に回す)。
  - 形式仕様で省略可とされる音符フィールド(`exp`/`aiExp`/`vibrato`/`singingSkill`/`phonemePositions`)の欠如。
  - オーディオトラックなど公開データモデルの `Track` に写像しないトラックの存在。
  - 歌唱パートに `notes` が無い場合は空の音符列として扱う。

### 3.2 続行可能な警告(`VprWarning`)

- **発音区間の重なり**(`code = "overlapping_notes"`): 歌唱トラックは単音想定で、同一パート内の音符の発音区間
  `[start_tick, start_tick + duration_tick)` が重なる音符ペアごとに1件 `VprWarning` を返す。`track_index`/`part_index`/
  `note_index`/`related_note_index`/`tick`(重なり開始位置)で位置を指す。検出のスコープは**同一パート内の重なり**で、
  異なるパート間にまたがる重なりの検出・ロケータ表現は初期スコープ外とし、必要が確認された時点で仕様化する。

### 3.3 未解釈データのロスレス保持

- 未解釈データ保持は、`Project/sequence.json` を JSON として読み込んだ結果を `VprProject.raw_sequence` に
  そのまま保持する範囲とする。公開モデルに写像しないトップレベルキー・歌唱トラック以外のトラック・対象外フィールドを含む。
- **保証しない範囲**: `Project/sequence.json` 以外の ZIP エントリ(`Project/Audio/*.wav` 等)・ZIP エントリ属性・
  JSON の空白/キー順/数値表記などの字句差・公開モデルの並び替えで失われる元順序・公開モデル各オブジェクトと
  `raw_sequence` 内 JSON オブジェクトとの対応関係は、公開 API として保証しない。
- 実際の往復(読み→書き→読み)検証と保持範囲の拡張(オブジェクト単位の `extras` 等)は `write` 実装時に行う(§4・§5)。

---

## 4. 書き出し(write)

- データモデル(2章)を vpr へ直列化する。
- 初期実装では読みを先行させ、書き出しは `song2vpr` 着手時に拡張する。書き出しは、対応する
  vpr で読み戻せる妥当な vpr を生成することを要件とする。

---

## 5. 形式レイアウトの正本

- vpr の具体的な直列化レイアウト(コンテナ構造・フィールド配置・バージョン差)は**形式仕様を正本**とし、
  本書では重複定義しない。本書は `vpr_io` の責務とデータモデル(2章)を定める。
- 形式仕様は [docs/specs/vpr/VPR_file_format.md](../docs/specs/vpr/VPR_file_format.md) を正本とする(MMD の VMD
  レイアウトを `docs/specs/vmd/VMD_file_format.md` が正本とし、`mmd_toolbox` がそれを実装するのと同じ扱い)。
  ロスレスの保持範囲(未解釈データの保持表現)は §3.3 に定める(`VprProject.raw_sequence` に `sequence.json`
  全体を保持。往復検証と範囲拡張は `write` 実装時)。

---

## 6. mmd_toolbox との関係・時間軸

- vpr はMMD形式でないため `mmd_toolbox` とは別モジュールとする。VMD側の入出力は `mmd_toolbox`、vpr側は
  `vpr_io`。
- vpr はテンポ・拍子に基づく時間表現(tick/拍)を持つ。一方 VMD 側は30fps基準(`mmd_toolbox` 規約)。
  時刻⇔フレーム/拍の変換は利用先(各CLI)が行い、`vpr_io` は vpr の時間表現とテンポ・拍子を正規化して
  渡すまでを担う(変換ポリシーは持たない)。
