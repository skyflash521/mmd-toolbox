# pmx — PMX読み取り・FK評価

PMXモデルファイルからボーン階層を読み取り、共通データモデルへ変換する。
さらに、VMDローカル姿勢をボーン階層へ適用する前方運動学(FK)評価を提供する。

バイナリレイアウトの正は `../../docs/specs/pmx/PMX仕様.txt`。本書ではバイト
レイアウトを重複記載せず、読み取り範囲・データモデル・規約・エラー・FK評価を定める。

---

## 1. 位置づけ

PMXのうちFK評価に必要なボーン情報だけを保持する読み取り専用パーサと、
VMDローカル姿勢をPMXボーン階層に適用する近似FK評価を提供する。
PMXの編集・書き出し、IK解決、付与親・物理演算の評価は対象外。

`pmx` はフォーマット層モジュール([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))で、
フォーマット層共通の設計原則(形式の事実のみを扱う・ロスレス・CLI非依存/無出力・構造化警告/エラー)は
[layering.md §3](../../docs/conventions/layering.md#3-フォーマット層共通の設計原則) を正本とする。

対応バージョンは PMX 2.0 / 2.1。非対応バージョンは読み取りを中断する。

---

## 2. 読み取り範囲

保持するもの:

- ボーン名(日本語・英語)とその生バイト。
- 親ボーンindex。
- ボーン基準位置(モデル原点からの座標)。
- 移動可否・回転可否。
- ボーンフラグの生値。

個数を読んで本体を読み飛ばすもの(保持しない):

- 頂点・面・テクスチャ・材質(ボーンへ到達するため)。
- モーフ・表示枠・剛体・ジョイント、および PMX 2.1 の SoftBody
  (ファイル構造の妥当性を検証するため)。

解釈しないもの(生フラグのみ保持):

- IKリンク詳細・付与親・軸制限・ローカル軸・外部親。

---

## 3. データモデル

### PmxModel

| フィールド | 内容 |
|---|---|
| `bones` | `PmxBone` のタプル(ファイル内の並び順) |
| `name_to_index` | ボーン名から `bones` のindexへの索引(同名は先出が優先) |
| `warnings` | `PmxWarning` のタプル |

### PmxBone

| フィールド | 内容 |
|---|---|
| `name` / `name_raw` | ボーン名(デコード文字列 / TextBufの生バイト) |
| `english_name` / `english_name_raw` | 英語ボーン名(同上) |
| `parent` | 親ボーンindex。非参照(負値)は `None` |
| `position` | 基準位置 `(x, y, z)` |
| `movable` | 移動可能フラグ(0x0004) |
| `rotatable` | 回転可能フラグ(0x0002) |
| `flags` | ボーンフラグ(16bit)の生値 |

### PmxWarning

| フィールド | 内容 |
|---|---|
| `code` | 事象コード |
| `message` | 説明 |
| `bone_index` | 関係するボーンindex(無ければ `None`) |

---

## 4. 規約

- 文字コードはPMXヘッダの指定(UTF-16LE または UTF-8)に従う。
  デコード不能バイトは置換文字で補い、生バイトを常に保持する。
- index参照値はPMXヘッダ指定の可変サイズ(1/2/4バイト)で読む。
- ボーン基準位置はモデル原点からの座標。
- `movable` / `rotatable` はボーンフラグから解決する。

---

## 5. エラーと警告

- **エラー(`PmxFormatError`)**: PMXマジック不正、未対応バージョン、未対応
  文字コード、indexサイズ不正(1/2/4以外)、追加UV数が範囲外(0〜4以外)、
  要素数が負、セクション長不足、未対応のウェイト変形方式・モーフ種類など、
  ファイル構造の異常。OSレベルのI/O失敗は標準例外を透過する。
- **警告(`PmxWarning`)**: FK近似で解釈しないボーンフラグ(IK・付与・物理後
  変形・外部親変形)を持つボーンは警告に残すが、読み取りは継続する。

---

## 6. API

```text
read_pmx(source: str | Path | bytes) -> PmxModel
```

`source` はPMXファイルのパス(`str` または `pathlib.Path`)、またはPMXバイト列。

---

## 7. FK評価

VMDボーンキーを指定フレームでサンプルしてローカル姿勢を作り、PMXボーン
階層に沿って前方運動学でワールド姿勢を評価する。IK・付与親・物理演算は
扱わない近似FKである。

### 7.1 データモデル

`LocalBonePose` / `WorldBonePose` はいずれも次のフィールドを持つ。

| フィールド | 内容 |
|---|---|
| `position` | 位置 `(x, y, z)` |
| `rotation` | クォータニオン `(x, y, z, w)` |

`LocalBonePose` はVMD由来のローカル移動量・回転量、`WorldBonePose` は
FK評価後のワールド位置・回転を表す。

### 7.2 評価規約

各ボーンのローカル変換とワールド変換は次で定める。

```text
base_offset       = bone.position - parent.position
local_translation = base_offset + (移動可能なら VMD位置)
local_rotation    = (回転可能なら VMD回転、でなければ単位回転)
world             = parent_world * translate(local_translation) * rotate(local_rotation)
```

- 親を持たないボーンはモデル原点を親とする。
- VMDにキーが無いボーンはローカル位置 `(0,0,0)`・単位クォータニオンを用いる。
- クォータニオンは `(x, y, z, w)` で扱い、正規化して適用する。
- 親が子より後ろのindexに置かれていても、ワールド姿勢を解決できる。

### 7.3 API

```text
sample_local_poses(model: PmxModel, bone_tracks: dict[str, list[BoneKey]], frame: int)
    -> tuple[LocalBonePose, ...]
evaluate_fk(model: PmxModel, local_poses: tuple[LocalBonePose, ...])
    -> tuple[WorldBonePose, ...]
evaluate_fk_range(model: PmxModel, bone_tracks: dict[str, list[BoneKey]], frames: range)
    -> list[tuple[WorldBonePose, ...]]
```

- `bone_tracks` はボーン名から `BoneKey` 列への対応。
- 返り値は `model.bones` と同じ並び順。
- `evaluate_fk_range` は `frames` の各フレームについて決定論的な結果を順に返す。
