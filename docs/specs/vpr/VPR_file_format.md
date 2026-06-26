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

```
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

### parts[](歌唱トラック)

| フィールド | 型 | 説明 |
|---|---|---|
| `name` | str | パート名 |
| `pos` | int | パート開始位置(プロジェクト絶対 tick) |
| `duration` | int | パート長(tick) |
| `notes[]` | list | 音符 |
| その他 | — | `styleName`/`aiVoice`/`controllers` 等 |

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
| `exp` / `aiExp` / `vibrato` / `singingSkill` | dict(**省略可**) | 表現パラメータ(ダイナミクス曲線等)。音符・サブフィールドにより有無が異なる |
| `phonemePositions[]` | list(**省略可**) | 音符内の音素別タイミング。音符により存在しない(実サンプルでは一部の音符のみ)。未設定値は `-2147483648`(INT_MIN) |

実サンプルの全音符に存在するフィールドは `pos`・`duration`・`number`・`lyric`・`phoneme`・`velocity` の6つ。
`exp`/`aiExp`/`vibrato`/`singingSkill`/`phonemePositions` は音符により有無が異なる(省略可)。

## 確証の状況

- **分解能 480:** ファイル内に分解能フィールドは無いが、実ファイルの音符位置(四分音符=480、1小節=1920)と
  整合し、独立実装([TheerapakG/vocaloid5tools](https://github.com/TheerapakG/vocaloid5tools) が `/ 480` で
  tick→拍へ換算)でも裏取りできる。別バージョンで異なる場合は本書を更新する。
- **note `pos` のパート相対:** 参照実装の [TheerapakG/vocaloid5tools](https://github.com/TheerapakG/vocaloid5tools)
  が `note.pos + part.pos` で絶対化しており、これに従いパート相対とみなす。解析に用いた実ファイルはいずれも
  `part.pos` = 0 のため、相対/絶対をローカルの実測だけでは区別できない(非ゼロ `part.pos` の実ファイルが
  得られれば最終確認できる)。`read` は `part.pos` を加算して絶対化する。
- **未解析の領域:** `exp`/`aiExp`/`vibrato`/`controllers` 等の表現パラメータの内部構造、オーディオトラックの
  詳細、`Project/sequence.json` 以外の ZIP エントリ(`Project/Audio/*.wav` 等)は、本書で詳細レイアウトを
  解析していない(確証が低い領域)。実ファイルで確定でき次第、本書を更新する。
