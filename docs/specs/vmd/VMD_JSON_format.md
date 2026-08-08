# VMD-JSON 表現フォーマット仕様

VMD の内容を人間が読み書きできる JSON で表現し、VMD と JSON を相互変換するための、ツール非依存の
フォーマット。VMD→JSON と JSON→VMD の双方向変換に対応する。

VMD バイナリレイアウトの正本は [VMD_file_format.md](VMD_file_format.md) とする。本書は VMD を JSON へ
写すときの構造・命名・単位と、JSON から VMD へ戻すときの挙動を定める。

---

## 1. 目的と対象範囲

### 1.1 目的

VMD はバイナリ形式で、補間曲線は 64 バイト（カメラは 24 バイト）の生レイアウト、名前は
Shift-JIS の固定長バイト列として格納される。そのままでは目視・手編集・差分確認ができない。

本フォーマットは、VMD の**内容**を次の条件で JSON に写し、JSON から VMD へも戻せるようにする。

- 名前は cp932 からデコードした文字列で持つ
- 補間曲線は 0〜127 の制御点（X/Y/Z/回転）に展開して持つ
- 値は VMD 格納値の意味（単位・符号）をそのまま保ち、再解釈はしない

VMD→JSON と JSON→VMD の双方向変換に対応する。バイト等価の無損失往復は保証しないが、JSON→VMD は
MMD で開ける妥当な VMD を生成する（[§4](#4-双方向変換vmd--jsonの挙動)）。生成した JSON は閲覧・手編集・差分確認にも使える。

### 1.2 対象範囲

本書は **VMD 1ファイル分の内容**（ヘッダと各キーフレームセクション）を 1つの JSON オブジェクトで表す
構造と、その双方向変換を定める。VMD のキーフレーム内容そのものを対象とする。

### 1.3 命名・エンコード

- ファイルは UTF-8 の JSON。非 ASCII 文字（日本語の名前等）は `\uXXXX` エスケープせず UTF-8 のまま保持する。
- 整形は 2 スペースインデントを推奨する。
- キー名は本書のとおり固定とする。読み手は未知キーを黙って捨てず、警告できることが望ましい。
- 拡張子は `.vmd.json` を推奨する。

---

## 2. トップレベル構造

```json
{
  "header": { "vmd_version": 2, "model_name": "初音ミク" },
  "bone": [],
  "morph": [],
  "camera": [],
  "light": [],
  "self_shadow": [],
  "ik_property": [],
  "sections": { "self_shadow": true, "ik_property": true }
}
```

| キー | 型 | 説明 |
|---|---|---|
| `header` | object | [§2.1](#21-header)。 |
| `bone` | array | [§3.1](#31-boneボーンキー)。ボーンキー列。空配列可。 |
| `morph` | array | [§3.2](#32-morphモーフキー)。モーフキー列。空配列可。 |
| `camera` | array | [§3.3](#33-cameraカメラキー)。カメラキー列。空配列可。 |
| `light` | array | [§3.4](#34-light照明キー)。照明キー列。空配列可。 |
| `self_shadow` | array | [§3.5](#35-self_shadowセルフ影キー)。セルフ影キー列。空配列可。 |
| `ik_property` | array | [§3.6](#36-ik_propertyikプロパティキー)。IK・プロパティキー列。空配列可。 |
| `sections` | object | [§2.2](#22-sections)。旧版 VMD で省略された後方セクションの有無。 |

- 各セクション配列は VMD の格納順（ファイル上の出現順）で並べる。フレーム昇順は保証しない
  （元ファイルがフレーム順でない場合があるため）。
- 空のセクションも配列キー自体は省略せず `[]` を置く。

### 2.1 header

| キー | 型 | 説明 |
|---|---|---|
| `vmd_version` | int | 固定値 `2`（現行標準の v2。magic 由来。[VMD_file_format.md](VMD_file_format.md) Header）。 |
| `model_name` | string | モデル名（cp932 デコード済み）。カメラ系 VMD では `"カメラ・照明"` 等。 |

本フォーマットは v2 を対象とする。v1（`"Vocaloid Motion Data file"`）は `model_name` 長が資料間で
異なり仕様が曖昧（[VMD_file_format.md](VMD_file_format.md) Header）なため対象外とし、`vmd_version` は
常に `2` とする。

### 2.2 sections

旧版 VMD ではセルフ影以降のセクションが物理的に省略される（[VMD_file_format.md](VMD_file_format.md)
ファイル全体構造）。その有無を真偽で持つ。

| キー | 型 | 説明 |
|---|---|---|
| `self_shadow` | bool | セルフ影セクションが元ファイルに存在したか。`false` のとき `self_shadow` は `[]`。 |
| `ik_property` | bool | IK・プロパティセクションが元ファイルに存在したか。`false` のとき `ik_property` は `[]`。 |

省略はセルフ影セクション以降が末尾まとめて切れる形でのみ起き（[VMD_file_format.md](VMD_file_format.md)
ファイル全体構造）、セルフ影が無いのに IK・プロパティだけ在る並びは作れない。したがって
`self_shadow` が `false` のとき `ik_property` も必ず `false` とする。とり得る組は
`{true, true}` / `{true, false}` / `{false, false}` の3通りで、`{false, true}` は不正。

---

## 3. セクション別キー構造

座標・角度・色は VMD 格納値の順序・単位・符号をそのまま JSON 配列／数値にする。再解釈しない。

全セクションのキーの `frame` は非負整数とする（[VMD_file_format.md](VMD_file_format.md) で各 `frame` は
`uint32`）。各表では `frame` の非負制約を個別に再掲しない。

### 3.1 bone（ボーンキー）

```json
{
  "name": "センター",
  "frame": 0,
  "position": [0.0, 0.0, 0.0],
  "rotation": [0.0, 0.0, 0.0, 1.0],
  "interpolation": {
    "X": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 },
    "Y": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 },
    "Z": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 },
    "R": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 }
  },
  "physics_off": false
}
```

| キー | 型 | 説明 |
|---|---|---|
| `name` | string | ボーン名（cp932 デコード済み）。 |
| `frame` | int | フレーム番号。 |
| `position` | number[3] | 位置 `[X, Y, Z]`。 |
| `rotation` | number[4] | 回転クォータニオン `[X, Y, Z, W]`。 |
| `interpolation` | object | 4チャンネル `X`/`Y`/`Z`/`R`（位置3軸＋回転）の制御点。各 `{x1, y1, x2, y2}` は 0〜127 の整数。4点ベジェ `(0,0),(x1,y1),(x2,y2),(127,127)`。 |
| `physics_off` | bool | （任意）そのキーで物理演算 OFF が指定されていたか。 |

- `interpolation` の制御点は、物理フラグで上書きされ得る生バイトではなく、シフトコピー側から復元した
  本来値を持つ（[VMD_file_format.md](VMD_file_format.md) 補間曲線・物理演算フラグ）。
- `physics_off` は物理フラグ（生バイト Byte[2], Byte[3] の上書き）を論理値として写したもの。元ファイルで
  物理 OFF 指定が無いキーでは省略してよい（既定 `false`）。
- チャンネル名 `R` は回転チャンネルを指す（バイナリ仕様の補間チャンネル名に準拠）。

### 3.2 morph（モーフキー）

```json
{ "name": "まばたき", "frame": 30, "weight": 1.0 }
```

| キー | 型 | 説明 |
|---|---|---|
| `name` | string | モーフ名（cp932 デコード済み）。 |
| `frame` | int | フレーム番号。 |
| `weight` | number | ウェイト（通常 0.0〜1.0、格納値そのまま）。 |

### 3.3 camera（カメラキー）

```json
{
  "frame": 0,
  "distance": -45.0,
  "position": [0.0, 10.0, 0.0],
  "angle": [0.0, 0.0, 0.0],
  "interpolation": {
    "x":        { "ax": 20, "bx": 107, "ay": 20, "by": 107 },
    "y":        { "ax": 20, "bx": 107, "ay": 20, "by": 107 },
    "z":        { "ax": 20, "bx": 107, "ay": 20, "by": 107 },
    "rotation": { "ax": 20, "bx": 107, "ay": 20, "by": 107 },
    "distance": { "ax": 20, "bx": 107, "ay": 20, "by": 107 },
    "fov":      { "ax": 20, "bx": 107, "ay": 20, "by": 107 }
  },
  "fov": 30,
  "perspective": true
}
```

| キー | 型 | 説明 |
|---|---|---|
| `frame` | int | フレーム番号。 |
| `distance` | number | カメラとカメラ中心の距離（負値で後方。格納値そのまま）。 |
| `position` | number[3] | カメラ中心の位置 `[X, Y, Z]`。 |
| `angle` | number[3] | 角度 `[X, Y, Z]`（ラジアン）。 |
| `interpolation` | object | 6チャンネル `x`/`y`/`z`/`rotation`/`distance`/`fov` の制御点。各 `{ax, bx, ay, by}`（始点X, 終点X, 始点Y, 終点Y）。 |
| `fov` | int | 視野角（度、整数）。 |
| `perspective` | bool | パースペクティブフラグ。`true`=ON（格納値 0）、`false`=OFF（格納値 1）。 |

- カメラのキーは「角度」であり「回転」ではない。クォータニオンではなくラジアン3成分。
- `perspective` は格納値（0=ON, 1=OFF）を真偽へ写す。0↔true の対応に注意。

### 3.4 light（照明キー）

```json
{ "frame": 0, "color": [0.6, 0.6, 0.6], "position": [-0.5, -1.0, 0.5] }
```

| キー | 型 | 説明 |
|---|---|---|
| `frame` | int | フレーム番号。 |
| `color` | number[3] | 色 `[R, G, B]`（各 0.0〜1.0）。 |
| `position` | number[3] | 照明の方向ベクトル `[X, Y, Z]`。 |

### 3.5 self_shadow（セルフ影キー）

```json
{ "frame": 0, "mode": 1, "distance": 0.089 }
```

| キー | 型 | 説明 |
|---|---|---|
| `frame` | int | フレーム番号。 |
| `mode` | int | セルフ影モード（0=OFF, 1=モード1, 2=モード2）。 |
| `distance` | number | 影距離パラメータ（格納値そのまま）。UI 上の距離値とは反転関係（[VMD_file_format.md](VMD_file_format.md) Self Shadow）。本書では再解釈せず格納値を持つ。 |

### 3.6 ik_property（IK・プロパティキー）

```json
{
  "frame": 0,
  "display": 1,
  "ik_bones": [
    { "name": "左足ＩＫ", "enable": 1 },
    { "name": "右足ＩＫ", "enable": 0 }
  ]
}
```

| キー | 型 | 説明 |
|---|---|---|
| `frame` | int | フレーム番号。 |
| `display` | int | モデル表示フラグ（0=非表示, 1=表示）。 |
| `ik_bones` | array | IK ボーンの有効/無効列。 |
| `ik_bones[].name` | string | IK ボーン名（cp932 デコード済み、最大20バイト由来）。 |
| `ik_bones[].enable` | int | IK 有効フラグ（0=無効, 1=有効）。 |

---

## 4. 双方向変換（VMD ⇔ JSON）の挙動

VMD→JSON と JSON→VMD の双方向に対応する。閲覧・手編集が成り立つよう値を可読化するため、
VMD→JSON→VMD でバイト等価には**ならない**。代わりに、JSON→VMD は MMD で開ける**妥当な VMD**を生成する。

### 4.1 保持される情報（VMD→JSON で失われない）

- すべてのキーのフレーム・数値（位置・回転・ウェイト・距離・角度・色・視野角・パースペクティブ）
- 補間曲線の制御点（X/Y/Z/R およびカメラ6チャンネル）
- 名前の文字列（cp932 でデコード可能な範囲）
- 各セクションの有無（[§2.2](#22-sections)）とキーの格納順

### 4.2 再構成される情報（JSON→VMD で正規化される）

名前のフィールド長は次のとおりで（[VMD_file_format.md](VMD_file_format.md) 各セクション）、書き出しは
cp932 エンコード結果をこの長さに収める。

| 名前 | フィールド長 |
|---|---|
| `model_name` | 20 バイト |
| ボーン名・モーフ名 | 各 15 バイト |
| IK ボーン名 | 20 バイト |

| 項目 | 往復時の扱い |
|---|---|
| 名前の固定長バイト列 | cp932 で再エンコードし、フィールド長に満たない分は null で埋める。エンコード結果がフィールド長を超える名前は書き出しエラーとし（切り詰めて名前を壊さない）、手編集側で短くする。ヌル終端**後**の残バイトは保持しない。 |
| 補間の 64／24 バイト生レイアウト | 制御点から再構成する。詰めパッド（`01`/`00`）は書き出し時の既定値で埋める（MMD バージョン依存のため固定値検証には使わない）。 |
| magic（ヘッダ） | v2 の magic（`"Vocaloid Motion Data 0002"`）を書く。 |
| 物理フラグ | `physics_off` が `true` のキーのみ Byte[2], Byte[3] を物理 OFF 値で書く。 |

### 4.3 非保証

- ヌル終端後の名前残バイト、補間の詰めパッドの元値、未知/将来セクションのバイト等価は保証しない。
- cp932 でデコードできないバイトを含む名前は置換文字になり、往復で元バイトへ戻らない。
  バイト等価の往復が要るときは VMD バイナリを直接扱うこと。

---

## 5. 完全な最小例

```json
{
  "header": { "vmd_version": 2, "model_name": "初音ミク" },
  "bone": [
    {
      "name": "センター",
      "frame": 0,
      "position": [0.0, 0.0, 0.0],
      "rotation": [0.0, 0.0, 0.0, 1.0],
      "interpolation": {
        "X": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 },
        "Y": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 },
        "Z": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 },
        "R": { "x1": 20, "y1": 20, "x2": 107, "y2": 107 }
      }
    }
  ],
  "morph": [
    { "name": "まばたき", "frame": 30, "weight": 1.0 }
  ],
  "camera": [],
  "light": [],
  "self_shadow": [],
  "ik_property": [],
  "sections": { "self_shadow": true, "ik_property": true }
}
```
