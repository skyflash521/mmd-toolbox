# vpr ファイルフォーマット仕様

> **注意:** vpr(VOCALOID プロジェクトファイル)に公式の公開仕様書は無い。本ドキュメントは実ファイルの
> 解析(リバースエンジニアリング)による結果をまとめたもので、誤りを含む可能性がある。本書は vpr の
> 直列化レイアウト(形式の事実)を正とする。読み手モジュールの責務・公開データモデル・写像は本書の範囲外で、
> 読み手側(各モジュール仕様)が本書を参照して定める。

## 主要参照元

| 参照元 | 種別 | 用途 |
|---|---|---|
| 実 vpr ファイルの解析(VOCALOID6 で作成、`version` 6.5.1) | 実ファイル | レイアウト・フィールド・値の一次確認 |
| [0x24a/vpr-parser](https://github.com/0x24a/vpr-parser) | リバースエンジニアリング実装(Pydantic パーサー) | フィールド名・構造の裏取り |
| [TheerapakG/vocaloid5tools](https://github.com/TheerapakG/vocaloid5tools) | リバースエンジニアリング実装(vpr→MIDI 変換) | 分解能 480・テンポ(value/100)・note の絶対化(`note.pos + part.pos`)の裏取り |
| [3c1u/vsqx](https://github.com/3c1u/vsqx) | リバースエンジニアリング実装(.vpr/.vsqx シリアライザ) | 参考 |
| [VOCALOID6 6.3 アップデートノート](https://www.vocaloid.com/en/learn/ln6212/) | 公式ドキュメント | `aiExp.vibratoLeadingDepth`/`vibratoFollowingDepth` が Vibrato Tool の「Leading Depth」「Following Depth」であることの裏取り |

> **注:** パーサー実装は仕様が不明確な箇所を実装で補っている場合があるため、断定の根拠としては実ファイルとの
> 整合を優先し、実装は裏取りに用いる。

## 基本仕様

- **コンテナ:** ZIP アーカイブ。拡張子 `.vpr`。
- **文字エンコード:** 内部の JSON は UTF-8。
- **対応バージョン:** VOCALOID5 以降の vpr(`sequence.json` を持つ形式)。解析に用いた実ファイルは
  `version` = `{major: 6, minor: 5, revision: 1}`、`vender` = `Yamaha Corporation`。
- **時間軸:** tick 整数。**分解能(tick/四分音符)はファイル内に格納されず、VOCALOID の固定値 480 とする**
  (PPQ=480。例: 四分音符の `duration` = 480、1小節=4/4 で 1920 tick)。

## ZIP エントリ構成

```text
Project/sequence.json              ← プロジェクト本体(JSON)
Project/Audio/<uuid>.wav           ← オーディオトラックの実体(0個以上)
```

- プロジェクト本体は `Project/sequence.json` の 1 ファイル。読み手が解析するのはこれ。
- `Project/Audio/*.wav` はオーディオトラックの波形データ。

## sequence.json 構造

トップレベルキー: `version`、`vender`、`title`、`masterTrack`、`voices`、`tracks`。

### version

| フィールド | 型 | 説明 |
|---|---|---|
| `major` / `minor` / `revision` | int | vpr のバージョン |

### masterTrack

テンポ・拍子など曲全体の時間情報。

| フィールド | 型 | 説明 |
|---|---|---|
| `samplingRate` | int | サンプリングレート(Hz) |
| `tempo.events[]` | list | テンポイベント。各要素 `{pos: int(tick), value: int}`。**`value` は BPM×100**(例 `13600` = 136.00 BPM)。`pos` の昇順 |
| `timeSig.events[]` | list | 拍子イベント。各要素 `{bar: int, numer: int, denom: int}`。**位置は tick でなく小節番号 `bar`(0 始まり)** |
| `volume.events[]` | list | マスターボリューム |

### tracks[]

| フィールド | 型 | 説明 |
|---|---|---|
| `type` | int | **2 = 歌唱トラック**(`notes` を持つ)、**1 = オーディオトラック**(`wav`/`region` を持ち音符なし) |
| `name` | str | トラック名 |
| `parts[]` | list | パート(歌唱区間/オーディオ区間) |
| その他 | — | `color`/`busNo`/`volume`/`panpot`/`isMuted` 等 |

### voices[]

歌唱に使うボイスバンクの定義。パートはここへ識別子で参照する(下記 [`parts[]`](#parts歌唱トラック) の `aiVoice`)。

| フィールド | 型 | 説明 |
|---|---|---|
| `compID` | str | ボイスバンクの識別子。パートの `aiVoice.compID` と一致するものが対応する |
| `name` | str | ボイスバンク名(例 `HATSUNE_MIKU_V6_ORIGINAL`) |

### `parts[]`(歌唱トラック)

| フィールド | 型 | 説明 |
|---|---|---|
| `name` | str | パート名 |
| `pos` | int | パート開始位置(プロジェクト絶対 tick) |
| `duration` | int | パート長(tick) |
| `notes[]` | list | 音符 |
| `controllers[]` | list | パート単位の連続コントローラ曲線(下記の [parts[] の controllers[]](#parts-の-controllers)) |
| `aiVoice` | dict | 使うボイスバンクの参照。`compID`(str。[`voices[]`](#voices) の同じ値を持つ要素が定義)と `langIDs`(list。各要素 `{langID: int}`)を持つ |
| `styleName` | str | 歌い方スタイルの名前(例 `Silk`)。`voices[]` とは対応せず、`stylePresetID`(str)と対で持つ |

オーディオトラックのパートは `{name, pos, wav, region}` を持ち `notes` を持たない。

### notes[]

| フィールド | 型 | 説明 |
|---|---|---|
| `pos` | int | 音符開始位置。**パート相対 tick**(`part.pos` を加算して絶対化する) |
| `duration` | int | 音符長(tick) |
| `number` | int | ピッチ(MIDI ノート番号) |
| `lyric` | str | 表示歌詞 |
| `phoneme` | str | 音素列。**空白区切り**(例 `"k o"`、`"t th e l"`)。`phoneme.split()` で音素の並びになる |
| `velocity` | int | ベロシティ(0〜127) |
| `vibrato` | dict(**省略可**) | 音符ビブラート。構造は下記の [notes[] の vibrato](#notes-の-vibrato) |
| `aiExp` | dict(**省略可**) | VOCALOID:AI の表現パラメータ。ビブラート深さの `vibratoLeadingDepth`/`vibratoFollowingDepth`(実数。観測値は約 0.2〜1.0 で、上端 1.0 は直接観測、下端 0 は未観測)を含む。それ以外のキー(`pitchFine`・`pitchDriftStart`/`End`・`pitchScalingCenter`/`Origin`・`pitchTransitionStart`/`End`・`amplitudeWhole`/`Start`/`End`)は名前以外を解析していない |
| `isAiVibratoEnabled` | bool(**省略可**) | 音符直下の真偽値(`aiExp` の内側ではない)。観測した音符はすべて `true` で、`false` の例は未観測。意味は未確定 |
| `exp` / `singingSkill` | dict(**省略可**) | 表現パラメータ。内部構造は解析していない |
| `phonemePositions[]` | list(**省略可**) | 音符内の音素別タイミング。音符により存在しない(実サンプルでは一部の音符のみ)。未設定値は `-2147483648`(INT_MIN) |

読み手が全音符での存在を当てにしてよいのは `pos`・`duration`・`number`・`lyric`・`phoneme`・`velocity` の
6つで、`exp`/`aiExp`/`vibrato`/`isAiVibratoEnabled`/`singingSkill`/`phonemePositions` は省略可として扱う
(欠落する音符がある前提で読む)。解析に用いた実ファイルでは、このうち `phonemePositions` だけが一部の音符に
しか無く、残りは全音符に存在した。

#### notes[] の vibrato

音符ごとのビブラート設定。VOCALOID エディタの Vibrato Tool が書き出す値にあたる。

| フィールド | 型 | 説明 |
|---|---|---|
| `type` | int | ビブラートの種別。**`0` 以外は未観測** |
| `duration` | int | ビブラート区間長(tick)。**`0` はビブラート無し** |
| `depths[]` | list(**省略可**) | 深さの自動化曲線。各要素 `{pos: int, value: int}` |
| `rates[]` | list(**省略可**) | 速さの自動化曲線。各要素 `{pos: int, value: int}` |

- `depths`/`rates` は**キー自体が存在しない音符がある**(明示的な JSON `null` ではない)。`duration > 0` でも
  両方とも欠落する音符、`rates` だけが欠落する音符のいずれも実ファイルで確認した。
- `depths`/`rates` の `value` の観測値域は **0〜127**(上下端とも実ファイルで直接観測)。最頻値はどちらも
  `64` で、`dynamics` と同じく値域中央に集中する。
- `depths`/`rates` の `pos` は **`0` の単一点しか観測されない**(実質は音符ごとの定数)。複数の制御点を持つ
  曲線は未観測のため、`pos` の原点(音符始端かビブラート区間始端か)は実ファイルから区別できない。
- `duration` は音符の `duration` 以下(違反は未観測)。区間が音符の**末尾側**に付くという解釈については
  下記の [確証の状況](#確証の状況) を参照。

### parts[] の controllers[]

パート単位の連続パラメータ自動化(オートメーション)曲線。各要素が1本の曲線。

| フィールド | 型 | 説明 |
|---|---|---|
| `name` | str | コントローラ名(下記の観測値) |
| `events[]` | list | 曲線の制御点。各要素 `{pos: int, value: int}`。`pos` は **パート相対 tick**(`part.pos` を加算して絶対化、notes と同じ)、`value` は生の整数値。ファイル上の `pos` は必ずしも昇順でない |

観測されたコントローラ名と値域(実ファイル解析。ボイスバンク/エディタ版により出現するものが異なる):

| name | 値域(観測) | 備考 |
|---|---|---|
| `dynamics` | 0〜127(中立 64) | VOCALOID の DYN(強弱・音量)。標準ボイスバンクで出現 |
| `s5Expression` | 0〜63(観測) | VOCALOID5 系/互換ボイスバンク由来の表情量。`dynamics` の代わりに出現する例を確認 |
| `character` / `s5Character` | 約 −4〜13(観測) | 声色 |
| `pitchBend` / `breathiness` / `brightness` / `clearness` / `portamento` / `growl` | — | ピッチ・息・音色等(実サンプルでは定数のことが多い) |

- 曲線は曲全体を被覆するとは限らない(部分区間だけ制御点を持つ例を確認。被覆外の値の扱いは読み手側の解釈)。
- 各コントローラが表す意味の確定・値域の正規化・他パラメータへの写像は本書の範囲外(読み手側が定める)。

## 確証の状況

- **分解能 480:** ファイル内に分解能フィールドは無いが、実ファイルの音符位置(四分音符=480、1小節=1920)と
  整合し、独立実装([TheerapakG/vocaloid5tools](https://github.com/TheerapakG/vocaloid5tools) が `/ 480` で
  tick→拍へ換算)でも裏取りできる。別バージョンで異なる場合は本書を更新する。
- **note `pos` のパート相対:** 参照実装の [TheerapakG/vocaloid5tools](https://github.com/TheerapakG/vocaloid5tools)
  が `note.pos + part.pos` で絶対化しており、これに従いパート相対とみなす。解析に用いた実ファイルはいずれも
  `part.pos` = 0 のため、相対/絶対をローカルの実測だけでは区別できない(非ゼロ `part.pos` の実ファイルが
  得られれば最終確認できる)。`read` は `part.pos` を加算して絶対化する。
- **controllers の event `pos` のパート相対:** note `pos` と同様に `part.pos` を加算して絶対化する(同じ tick
  空間。実ファイルでも controller event の tick 範囲が note の tick 範囲と整合する)。`part.pos` = 0 のため相対/
  絶対の最終区別は非ゼロ `part.pos` の実ファイル待ち(上の note `pos` と同じ状況)。コントローラの基本レイアウト
  (`name`・events の `pos`/`value`)は解析済みだが、観測したコントローラ名の網羅・各名前が表す意味・被覆外の値の
  解釈は確定していない。
- **`vibrato` の区間位置:** `vibrato.duration` は音符の `duration` 以下で、区間が音符の**末尾側**
  (`note.duration − vibrato.duration` から音符終端まで)に付くと解釈する。区間の開始位置を明示する
  オフセットフィールドは存在せず、この解釈は「観測した全音符が矛盾なく収まる」という間接証拠と、
  ビブラートが音符の後半に自動で付く VOCALOID の UI 挙動に基づく。直接の裏取りは取れていない。
- **`vibratoLeadingDepth`/`vibratoFollowingDepth` の意味:** VOCALOID 公式の 6.3 アップデートノートが、
  Vibrato Tool へ追加した「Leading Depth」「Following Depth」として説明している(音の終わりに向けて
  ビブラートを顕著にする表現)。実ファイルでの観測値は約 0.2〜1.0 に散り、`0.5` が多数派。上端 `1.0` は
  直接観測したが下端 `0` は未観測で、値域が 0〜1 であることと、`0.5` が「変化なし」の中立値であることは、
  いずれも公式文書に明記が無く観測パターンからの推定にとどまる。
- **未解析の領域:** `exp`/`singingSkill` および `aiExp` のビブラート深さ2キー以外の内部構造、
  `isAiVibratoEnabled` の意味、`aiVoice.langIDs` の値の意味、`stylePresetID` の値の意味、
  オーディオトラックの詳細、`Project/sequence.json` 以外の ZIP エントリ
  (`Project/Audio/*.wav` 等)は、本書で詳細レイアウトを解析していない(確証が低い領域)。実ファイルで
  確定でき次第、本書を更新する。
