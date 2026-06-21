# VMD ファイルフォーマット仕様

> **注意:** VMD フォーマットは公式の仕様書が存在しない。本ドキュメントはリバースエンジニアリングによる解析結果をまとめたものであり、誤りを含む可能性がある。

## 主要参照元

| 参照元 | 種別 | ライセンス |
|--------|------|-----------|
| [libmmd wiki](https://github-wiki-see.page/m/Antonio225t/libmmd/wiki/VMD) | リバースエンジニアリング仕様書 | — |
| [Nuthouse01/PMX-VMD-Scripting-Tools](https://github.com/Nuthouse01/PMX-VMD-Scripting-Tools) | Python ツール群 + README 仕様説明 | — |
| [MikuMikuDance Wiki — VMD file format](https://mikumikudance.fandom.com/wiki/VMD_file_format) | コミュニティ仕様書 | — |
| [babylon-mmd — Introduction to VMD and VPD](https://noname0310.github.io/babylon-mmd/docs/reference/understanding-mmd-behaviour/introduction-to-vmd-and-vpd/) | 解説ドキュメント | — |
| [針金P — VMDメモ](https://hariganep.seesaa.net/article/201103article_1.html) | 補間パラメータ 64 bytes / カメラ補間 24 bytes の解析 | — |
| [Olyutorskii — VMDモーションファイルの仕様変更について](https://olyutorskii.wordpress.com/2013/05/21/newvmd/) | 詰めパッドのバージョン差異・Ver7.40 での IK 枠追加の解析 | — |
| [oguna/MMDFormats](https://github.com/oguna/MMDFormats) — `MikuMikuFormats/Vmd.h` | C++ パーサー実装（参考） | CC0 1.0 |
| [takahirox/mmd-parser](https://github.com/takahirox/mmd-parser) — `src/Parser.js` | JavaScript パーサー実装（参考） | MIT |

> **注:** `oguna/MMDFormats` と `takahirox/mmd-parser` はパーサー実装として参考にしているが、仕様が不明確な箇所を当時の実装で補っている部分があるため、仕様確定の根拠としては扱わない。

---

## 基本仕様

- **エンディアン:** リトルエンディアン
- **文字エンコード:** Shift-JIS（ボーン名・モーフ名・モデル名）
- **構造:** 各セクションはキーフレーム数 (uint32) + キーフレーム配列の形式
- **時間軸:** VMD ファイル自体はフレームレート値を保持せず、各キーフレームはフレーム番号 (`frame`) のみを持つ
- **運用上の扱い:** 1つの VMD にモデル系（Bone/Morph/IK）とカメラ系（Camera/Light/Self Shadow）の両方を格納できるが、MMD では `model_name` が `カメラ・照明` の場合は主にカメラ系として、それ以外では主にモデル系として扱われる実装例がある。

---

## ファイル全体構造

```
[Header]
[Bone Keyframes]
[Morph Keyframes]
[Camera Keyframes]
[Light Keyframes]
[Self Shadow Keyframes]   ← 省略される場合あり（旧バージョン）
[IK/Property Keyframes]  ← 省略される場合あり（旧バージョン）
```

---

## Header

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| magic | char[30] | 30 | マジックバイト。後述のバージョン識別に使用 |
| model_name | char[?] | **v1: 10 or 15, v2: 20** | モデル名（Shift-JIS、ヌル終端） |

**magic の値によるバージョン識別:**

| magic 文字列 | バージョン | model_name サイズ | 備考 |
|-------------|-----------|-----------------|------|
| `"Vocaloid Motion Data file"` | v1 | 10 または 15 bytes | MikuMikuDance 1.30 以前（非 Multi-Model 版） |
| `"Vocaloid Motion Data 0002"` | v2 | 20 bytes | Multi-Model Edition 以降（現行標準） |

> **model_name サイズの注意:** v1 のサイズについて資料間で差異がある（libmmd wiki: 15 bytes、Nuthouse01: 10 bytes）。実装上の裏付けは 10 bytes の方が強い（Nuthouse01 は実際に v1 ファイルを 10 bytes でパースしているのに対し、15 bytes は libmmd wiki の記述のみ）。ただし v1 ファイルを扱う場合は単一値として断定せず、実ファイルの後続オフセット・骨フレーム数・ファイル長との整合で妥当性確認すること。v2 の 20 bytes は複数資料で一致している。  
> `oguna/MMDFormats` および `takahirox/mmd-parser` は model_name を常に 20 bytes 読んでおり、v1 を正しく処理できない可能性がある。

---

## Bone Keyframes（ボーンキーフレーム）

先頭に uint32 でキーフレーム数を格納し、その数だけ以下の構造が連続する。

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| bone_name | char[15] | 15 | ボーン名（Shift-JIS、ヌル終端） |
| frame | uint32 | 4 | フレーム番号 |
| position | float[3] | 12 | 位置 (X, Y, Z) |
| rotation | float[4] | 16 | 回転クォータニオン (X, Y, Z, W) |
| interpolation | uint8[64] | 64 | 補間曲線パラメータ（後述） |

**1キーフレームの合計サイズ:** 111 bytes

### 補間曲線（interpolation, 64 bytes）

4つの補間軸（X位置・Y位置・Z位置・回転）それぞれについて、4点ベジェ曲線 `(0,0), (x1,y1), (x2,y2), (127,127)` の制御点座標 (x1, y1, x2, y2) を 0〜127 の整数で表す。  
座標系は 128×128 グリッド（左下原点）。

記法: `{軸}_{座標}{制御点番号}` — 例: `X_x1` = X軸の第1制御点のx座標、`R_y2` = 回転軸の第2制御点のy座標。

**全 64 バイトのレイアウト（各 1 byte）:**

```
Byte[ 0] = X_x1   Byte[ 1] = Y_x1   Byte[ 2] = Z_x1   Byte[ 3] = R_x1
Byte[ 4] = X_y1   Byte[ 5] = Y_y1   Byte[ 6] = Z_y1   Byte[ 7] = R_y1
Byte[ 8] = X_x2   Byte[ 9] = Y_x2   Byte[10] = Z_x2   Byte[11] = R_x2
Byte[12] = X_y2   Byte[13] = Y_y2   Byte[14] = Z_y2   Byte[15] = R_y2

Byte[16] = Y_x1   Byte[17] = Z_x1   Byte[18] = R_x1   Byte[19] = X_y1
Byte[20] = Y_y1   Byte[21] = Z_y1   Byte[22] = R_y1   Byte[23] = X_x2
Byte[24] = Y_x2   Byte[25] = Z_x2   Byte[26] = R_x2   Byte[27] = X_y2
Byte[28] = Y_y2   Byte[29] = Z_y2   Byte[30] = R_y2   Byte[31] = 01

Byte[32] = Z_x1   Byte[33] = R_x1   Byte[34] = X_y1   Byte[35] = Y_y1
Byte[36] = Z_y1   Byte[37] = R_y1   Byte[38] = X_x2   Byte[39] = Y_x2
Byte[40] = Z_x2   Byte[41] = R_x2   Byte[42] = X_y2   Byte[43] = Y_y2
Byte[44] = Z_y2   Byte[45] = R_y2   Byte[46] = 01     Byte[47] = 00

Byte[48] = R_x1   Byte[49] = X_y1   Byte[50] = Y_y1   Byte[51] = Z_y1
Byte[52] = R_y1   Byte[53] = X_x2   Byte[54] = Y_x2   Byte[55] = Z_x2
Byte[56] = R_x2   Byte[57] = X_y2   Byte[58] = Y_y2   Byte[59] = Z_y2
Byte[60] = R_y2   Byte[61] = 01     Byte[62] = 00     Byte[63] = 00
```

補間パラメータ本体は先頭 16 バイト（Byte[0]〜Byte[15]）に格納されている。Byte[16] 以降は先頭シーケンスを 1 バイトずつ左シフトしたコピーで、末尾には詰めパッド（上表の `01`, `00`）が埋められる。

> **詰めパッドは MMD バージョン依存:** 上表の `01` は MMD Ver7.39dot までの旧仕様の値。Ver7.39.x64 以降の MMD は詰めパッド先頭に `00` を書き込む（Olyutorskii の解析による）。固定値としてバリデーション等に使ってはならない。
>
> **物理演算フラグ:** ボーンキーフレーム単位の物理演算 ON/OFF 設定は、**先頭 16 バイト内の Byte[2], Byte[3]**（Z_x1, R_x1 の位置）を上書きする形で格納される（`(99, 15)` = 物理 OFF。Nuthouse01 の実装解析による）。このため物理フラグ付きのファイルでは「先頭 16 バイトだけ読めば十分」は成立せず、本来の Z_x1, R_x1 はシフトコピー側（Byte[17], Byte[18]）から復元する必要がある。再エンコード時は元の 64 バイトをそのまま保持することが安全。

`oguna/MMDFormats` では `char interpolation[4][4][4]` として宣言されている。

---

## Morph Keyframes（モーフキーフレーム）

先頭に uint32 でキーフレーム数を格納し、その数だけ以下の構造が連続する。

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| morph_name | char[15] | 15 | モーフ名（Shift-JIS、ヌル終端） |
| frame | uint32 | 4 | フレーム番号 |
| weight | float | 4 | モーフウェイト（0.0〜1.0） |

**1キーフレームの合計サイズ:** 23 bytes

---

## Camera Keyframes（カメラキーフレーム）

先頭に uint32 でキーフレーム数を格納し、その数だけ以下の構造が連続する。

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| frame | uint32 | 4 | フレーム番号 |
| distance | float | 4 | カメラとカメラ中心の間の距離（負値で後方） |
| position | float[3] | 12 | カメラ中心の位置 (X, Y, Z) |
| rotation | float[3] | 12 | 角度（ラジアン, X, Y, Z） |
| interpolation | uint8[24] | 24 | 補間曲線パラメータ（後述） |
| fov | uint32 | 4 | 視野角（度、整数） |
| perspective | uint8 | 1 | パースペクティブフラグ（0: ON, 1: OFF） |

**1キーフレームの合計サイズ:** 61 bytes

> **実装間の差異:** `oguna/MMDFormats` は fov を `float` として読み込み、後続 3 バイトを `unknown` として扱う（合計 63 bytes）。`takahirox/mmd-parser` は fov を `uint32`、perspective を `uint8` として読み込む（合計 61 bytes）。`uint32` 解釈の方が意味的に整合しており、後者が正確と考えられる。

### カメラ補間曲線（interpolation, 24 bytes）

6軸それぞれについて、ベジェ曲線の制御点 (ax, bx, ay, by) を 4 bytes で格納する。各値は 1 byte で保持され、実用上は 0〜127 の範囲で扱われることが多い（資料によっては s8 と表記される）。  
制御点の格納順は **`ax, bx, ay, by`（始点X, 終点X, 始点Y, 終点Y）**。

**軸の順序（libmmd wiki のオフセットに基づく）:**

```
Byte[0..3]   = [ax, bx, ay, by] の X位置軸
Byte[4..7]   = [ax, bx, ay, by] の Y位置軸
Byte[8..11]  = [ax, bx, ay, by] の Z位置軸
Byte[12..15] = [ax, bx, ay, by] の 回転軸
Byte[16..19] = [ax, bx, ay, by] の 距離軸
Byte[20..23] = [ax, bx, ay, by] の 視野角軸
```

---

## Light Keyframes（照明キーフレーム）

先頭に uint32 でキーフレーム数を格納し、その数だけ以下の構造が連続する。

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| frame | uint32 | 4 | フレーム番号 |
| color | float[3] | 12 | 色 (R, G, B)、各値 0.0〜1.0 |
| position | float[3] | 12 | 照明の方向ベクトル (X, Y, Z) |

**1キーフレームの合計サイズ:** 28 bytes

---

## Self Shadow Keyframes（セルフ影キーフレーム）

先頭に uint32 でキーフレーム数を格納し、その数だけ以下の構造が連続する。  
旧バージョンの VMD ではこのセクション以降が省略される場合がある。

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| frame | uint32 | 4 | フレーム番号 |
| mode | uint8 | 1 | セルフ影モード（0x00: OFF, 0x01: モード1, 0x02: モード2） |
| distance | float | 4 | 影距離パラメータ。`格納値 = (10000 - 距離) / 100000`（格納範囲は 0.0〜0.1 で、UI 上の距離値とは反転関係。blender mmd_tools / Nuthouse01 の実装で一致） |

**1キーフレームの合計サイズ:** 9 bytes

> **`oguna/MMDFormats` の `unknown2` について:** 同実装では Light 読み込み後に 4 バイトを `unknown2` として読み飛ばしているが、これは Self Shadow セクションの **キーフレーム数 (uint32)** に相当する。セルフ影フレームが 0 件の VMD では結果的に正しく動くが、セルフ影を含む VMD では IK 以降のオフセットが壊れる。正しいセクション順序は `Light → Self Shadow → IK/Property` である。

---

## IK / Property Keyframes（IK・プロパティキーフレーム）

先頭に uint32 でキーフレーム数を格納し、その数だけ以下の構造が連続する。  
Self Shadow セクション同様、旧バージョンでは省略される場合がある。

### キーフレーム本体

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| frame | uint32 | 4 | フレーム番号 |
| display | uint8 | 1 | モデルの表示フラグ（0: 非表示, 1: 表示） |
| ik_count | uint32 | 4 | IK ボーンの数 |

### IK ボーン情報（ik_count 個連続）

| フィールド | 型 | サイズ (bytes) | 説明 |
|-----------|-----|---------------|------|
| ik_name | char[20] | 20 | IK ボーン名（Shift-JIS、ヌル終端） |
| enable | uint8 | 1 | IK 有効フラグ（0: 無効, 1: 有効） |
