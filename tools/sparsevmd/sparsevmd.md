# sparsevmd 仕様書

密なVMDキーフレームを疎なキーフレームと補間曲線へ変換するCLIツール

実装言語: Python 3.12+
依存: vmd(同リポジトリのフォーマット層ライブラリ、../../libs/vmd/vmd.md), numpy, scipy, click または argparse

---

## 1. 概要

### 1.1 目的
shakevmdの出力VMDやモーションキャプチャー由来VMDのように、
連続フレームにキーフレームが存在する高密度モーションを、
MMD互換の疎なキーフレームと補間曲線へ変換する。

目的は以下の3つ:
- MMD上で扱いやすいキー数に減らす
- 再生結果を指定誤差以内に保つ
- 補間曲線を使った自然な編集可能モーションに戻す

### 1.2 基本方針
- 対象は初期実装では **カメラキー** と **ボーンキー** とする。
  shakevmd出力のカメラVMD、モーションキャプチャー由来のボーンVMDの両方を扱う。
- モーフ/照明/セルフ影/IKキーは初期実装では無加工で透過する。
  モーフはVMD上に補間曲線を持たないため、別仕様として扱う。
- 入力モーションを30fps整数フレームでサンプリングし、出力VMDを同じフレームで
  再評価したときの誤差が許容値以下になるようにキーを削減する。
- 削減は「区間の両端キー + VMD補間曲線」で表現できるかを検証し、
  表現できない区間は自動的に分割する。最悪の場合は元の密なキー数を保持するため、
  近似不能なモーションでも破綻させない。
- カット、ワープ、瞬間的な姿勢切り替えは連続曲線でなまらせず、
  境界として保持する。
- 同一入力・同一引数からは常に同一出力。

### 1.3 非目標
- VMDの完全な可逆圧縮。対象セクションは再構築されるためバイト一致は保証しない
- モーションキャプチャーのノイズ除去、足接地補正、IK補正、姿勢推定の修正
- MMD/VMDに存在しない補間方式の導入
- 60fps化や任意fpsへのリサンプリング
- GUI

---

## 2. コマンドライン仕様

### 2.1 書式

    sparsevmd <入力ファイル名>.vmd [オプション]

### 2.2 入出力・対象

| 引数 | 型 / 書式 | デフォルト | 説明 |
|---|---|---|---|
| `<入力ファイル名>` | パス | (必須) | 入力VMD |
| `-o, --output` | パス | `<入力ファイル名>_sparse.vmd` | 出力先 |
| `--overwrite` | flag | off | 出力先の既存ファイルへの上書きを許可。未指定で出力先に既存ファイルがあるとエラー |
| `--target` | `camera` / `bone` / `all` | `all` | 削減対象セクション。`all` は camera + bone |
| `--bone NAME` | NAME(反復可) | なし | 指定ボーンのみ処理。未指定なら全ボーン |
| `--bone-glob PATTERN` | GLOB(反復可) | なし | globに一致するボーンを処理対象に追加 |
| `--bone-group GROUP` | GROUP(反復可) | なし | よく使うボーン集合を処理対象に追加 |
| `--bone-file PATH` | パス | なし | ボーン選択ルールをUTF-8テキストから読み込む |
| `--exclude-bone NAME` | NAME(反復可) | なし | 指定ボーンを処理対象から除外 |
| `--exclude-bone-glob PATTERN` | GLOB(反復可) | なし | globに一致するボーンを処理対象から除外 |
| `--exclude-bone-group GROUP` | GROUP(反復可) | なし | よく使うボーン集合を処理対象から除外 |
| `--list-bones` | flag | off | 入力VMD内のボーン名・キー数・選択状態を表示して終了 |
| `--range RANGE` | RANGE(反復可) | 全範囲 | 削減範囲。`START:END` 形式。START/ENDの一方は省略可 |

CLI値の共通書式:
- `反復可` は同じオプションを複数回指定することを意味する。
  カンマ区切りリストは受け付けない。
  例: `--bone センター --bone 上半身`、`--range 0:240 --range 300:420`
- `NAME` はシェルから渡されるUnicode文字列。
  VMD内のボーン名を固定長バイト列の先頭からNULまでCP932でデコードした表示名と
  完全一致・大文字小文字区別ありで照合する。glob/正規表現/部分一致は行わない。
  空文字列はエラー。デコード不能なボーン名は警告し、`--bone` /
  `--exclude-bone` の名前指定では一致不可とする。
- `GLOB` はPython `fnmatchcase` 相当のワイルドカード。
  `*` は0文字以上、`?` は1文字、`[abc]` は文字集合に一致する。
  シェルに展開されないよう、`--bone-glob '*指*'` のように引用して使う。
  正規表現は使わない。
- ボーン選択は include 後に exclude を適用する。
  include指定(`--bone` / `--bone-glob` / `--bone-group` / `--bone-file` 内のinclude)が
  1つもない場合は全ボーンをincludeしたものとして扱う。
  その後、`--exclude-bone` / `--exclude-bone-glob` / `--exclude-bone-group` /
  `--bone-file` 内のexcludeを引く。
- `--bone` と `--exclude-bone` の両方に同じ `NAME` が指定された場合はエラー
  (終了コード2)。`--bone` で指定した名前が入力VMDに存在しない場合もエラー。
  この判定は `--target all` でも適用し、§3.1のカメラのみ処理フォールバックより
  優先する。すなわち `--bone NAME` で明示した名前が(ボーンセクションが空の場合を
  含め)入力に存在しなければ、カメラの有無に関わらず引数エラー(終了コード2)とする。
  `--exclude-bone` のみで存在しない名前を指定した場合は警告して続行する。
  glob/groupが1件も一致しない場合は、その選択子について警告して続行する
  (その選択子は何も寄与しないものとして扱い、他の選択子の一致は活きる)。
  ただしこの警告は選択子単位の通知であり、最終的な選択結果の判定とは別段である。
  ボーンセクションにキーが存在するにもかかわらず、指定されたボーン選択ルールの
  include/exclude適用結果が(理由を問わず)0件になる場合は引数エラー(終了コード2)。
  したがって唯一のinclude選択子が不一致のglob/groupだった場合は、警告を出した上で
  最終結果0件として終了コード2になる。
  入力VMDに対象セクションのキー自体が存在しない場合は入力不正(終了コード1、§3.1)。
- `--target camera` とボーン選択オプションを同時指定した場合はエラー
  (終了コード2)。ただし `--list-bones` 指定時は検査モードとして扱い、
  `--target` に関係なくボーン一覧と選択状態を表示する。
  `--target all` ではボーン選択はboneセクションのみに作用し、
  cameraセクションは通常通り処理する。
- `--bone-file PATH` はUTF-8テキスト。空行と `#` で始まる行は無視する。
  各行は `name:NAME`、`glob:PATTERN`、`group:GROUP`、`exclude:name:NAME`、
  `exclude:glob:PATTERN`、`exclude:group:GROUP` のいずれか。
  接頭辞なしの行は `name:` として扱う。同じ選択子の重複は1つに正規化する。
- `GROUP` は以下の組み込みグループ名。各グループはGLOBの集合として定義する。
  グループは完全なボーン分類ではなく、MMD標準名・準標準名・よくある英語名を
  まとめて選びやすくするための便宜機能である。実際に一致したボーンは
  `--dry-run` / `--list-bones` で確認する。

| GROUP | 展開されるGLOB |
|---|---|
| `core` | `センター`, `グルーブ`, `全ての親`, `上半身*`, `下半身`, `首`, `頭`, `center`, `groove`, `root`, `upper body*`, `lower body`, `neck`, `head` |
| `arms` | `*肩*`, `*腕*`, `*ひじ*`, `*肘*`, `*手首*`, `*shoulder*`, `*arm*`, `*elbow*`, `*wrist*` |
| `legs` | `*足*`, `*脚*`, `*ひざ*`, `*膝*`, `*つま先*`, `*leg*`, `*knee*`, `*ankle*`, `*toe*` |
| `fingers` | `*指*`, `*finger*`, `*thumb*`, `*index*`, `*middle*`, `*ring*`, `*pinky*`, `*little*` |
| `ik` | `*IK*`, `*ＩＫ*`, `*ik*` |
| `mocap` | `core` + `arms` + `legs` + `fingers`。`ik` は含めない |

- `RANGE` は `START:END`、`START:`、`:END` のいずれか。
  START/ENDは0以上の10進整数フレーム番号で、両端を含む。
  `START > END` はエラー。省略されたSTART/ENDは、ボーン選択と `--target` を
  適用した後の対象トラック全体の最小/最大フレームに1回だけ展開する。
  トラックごとには展開しない。
  複数範囲はフレーム昇順に正規化し、展開後に1フレームでも重複したらエラー
  (終了コード2)。例: `10:20` と `20:30` は20が重複するためエラー。
  各トラックの実処理範囲は、このグローバル範囲と当該トラックの先頭/末尾フレームの
  積集合とする。積集合が空のトラックは無加工で保持し、dry-runに記録する。
  すべての対象トラックで積集合が空の場合も正常終了(終了コード0)とし、
  出力VMDを作る場合は対象セクションも含め元のキーを保持する。
  dry-runには「削減対象なし」と記録する。
- `flag` は値を取らない真偽オプション。同じflagを複数回指定しても1回指定と同じ。
- `パス` はローカルファイルパス。入力パスが存在しない/通常ファイルでない場合は
  引数エラー(終了コード2、読み込み前のパス検証で判定)。
  出力VMDの親ディレクトリが存在しない場合や、
  その他の書き込み失敗(`io.write_file` 等が送出する例外)は出力書き込み失敗
  (終了コード3、§9)。
  `--overwrite` は出力先に既存ファイルがあるときの上書き許可を制御する
  (入力パスと同一かどうかは問わない)。

`--range` 指定時、対象トラックの**範囲外キーは値・補間曲線ともに変更不可**で逐語保持する
(§6.3)。範囲端は必須キーとして範囲内に保持する。範囲外への境界キー追加や範囲外キーの
補間曲線書き換えは行わない。範囲開始キー(範囲内)の到達側曲線を手前の範囲外区間の動きに
合わせて再フィットした場合は、`--dry-run` と verbose ログ・レポート(診断 `seam_rewrites`)で
報告する。

### 2.3 品質プリセット

| 引数 | 型 / 書式 | デフォルト | 説明 |
|---|---|---|---|
| `--preset NAME` | `precise` / `balanced` / `aggressive` | `balanced` | 品質プリセット |

プリセットの意味:
- `precise`: 誤差を小さくし、キー削減率より忠実度を優先
- `balanced`: 標準。見た目の差を抑えつつキー数を削減
- `aggressive`: キー削減率を優先。モーションの細部は丸くなる

個別の許容誤差オプションが明示された場合はプリセットより優先する。

### 2.4 許容誤差

列 `precise` / `balanced` / `aggressive` は各プリセット選択時のデフォルト値。
個別オプション明示時はその値が優先する(§2.3)。

| 引数 | 型 / 単位 | `precise` | `balanced` | `aggressive` | 説明 |
|---|---|---|---|---|---|
| `--bone-pos-tol` | MMD距離単位 | 0.005 | 0.01 | 0.05 | ボーン位置の最大許容誤差 |
| `--bone-rot-tol` | 度 | 0.05 | 0.10 | 0.50 | ボーン回転の最大角度誤差 |
| `--camera-pos-tol` | MMD距離単位 | 0.01 | 0.02 | 0.10 | カメラ中心位置の最大許容誤差 |
| `--camera-rot-tol` | 度 | 0.02 | 0.05 | 0.25 | カメラ回転の最大角度誤差 |
| `--camera-distance-tol` | MMD距離単位 | 0.01 | 0.02 | 0.10 | カメラ距離の最大許容誤差 |
| `--camera-fov-tol` | 度 | 0.50 | 0.50 | 1.00 | 視野角の最大許容誤差 |

許容誤差の意味(`0` の完全一致・float32 量子化誤差の扱い・視野角整数度の意味)は
[vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §2 を正とする。

各値は0以上の有限小数。`--camera-fov-tol` はVMDの視野角キーが整数度保存であるため
0.5以上でなければならない。0.5未満はエラー(終了コード2)。
NaN/Inf、負値、単位付き文字列はエラー(終了コード2)。

### 2.5 フィット制御

| 引数 | 型 | デフォルト | 説明 |
|---|---|---|---|
| `--max-segment-frames` | int | 無制限 | 出力キー間隔の sliding 上限。未指定なら上限なし(無制限)で、滑らかな長区間を長いまま残す。有限値を指定すると、機械的な等分点は出力キーにせず、確定キーから上限を超える非定数区間だけ分割する |
| `--min-segment-frames` | int | 1 | これ以上は分割しない最小区間長。1なら隣接キーを保持可能 |
| `--curve-mode` | `bezier` / `linear` | `bezier` | `linear` は補間曲線を線形固定してキー削減のみ行う |
| `--strict` | flag | off | 指定誤差を満たせない場合に高密度キー保持で続行せずエラー |

区間化の挙動(最小区間長の分割下限・最大区間長の sliding 上限・strict の扱い)は
[vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §3 を正とする。

`--min-segment-frames` は1以上の整数。`--max-segment-frames` は未指定なら上限なし(無制限)で、
指定する場合は1以上の整数。`--max-segment-frames` を指定したとき
`--min-segment-frames > --max-segment-frames` はエラー(終了コード2)。
`--strict` ありで誤差を満たせない場合は終了コード4とする。

### 2.6 カット・不連続制御

| 引数 | 書式 | デフォルト | 説明 |
|---|---|---|---|
| `--cut-threshold-camera` | `POS,ROT,DIST` | `5.0,20.0,5.0` | カメラの不連続検出。位置・角度・距離差のいずれかが超過したら境界 |
| `--cut-threshold-bone` | `POS,ROT` | `1.0,30.0` | ボーンの不連続検出。位置または角度差が超過したら境界 |
| `--cut-detect` | flag | on | 閾値による自動境界検出を有効化(既定 on)。`--no-cut-detect` の対の明示形 |
| `--no-cut-detect` | flag | off | 閾値による自動境界検出を無効化。`--cut-detect` の対 |
| `--keep-frame FRAME` | FRAME(反復可) | なし | 指定フレームを必ずキーとして保持 |

`POS,ROT,DIST` はカンマ区切りの0以上の有限小数3つ。
`POS,ROT` はカンマ区切りの0以上の有限小数2つ。
空白・単位は含めない。`POS` と `DIST` の単位はMMD距離単位、`ROT` の単位は度。
例: `--cut-threshold-camera 5.0,20.0,5.0`。
`FRAME` は0以上の10進整数フレーム番号。複数指定時の重複は1つに正規化する。
`FRAME` が全削減範囲の外にある場合は警告して無視する。
特定トラックの有効範囲外にある場合、そのトラックでは無視する。

不連続・境界処理(境界を `F-1`/`F` の間に置くこと、境界をまたぐ区間フィットを行わずジャンプを
保持すること)は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §7 を正とする。

### 2.7 レポート・運用

| 引数 | 説明 |
|---|---|
| `--dry-run` | 出力VMDを書かずに統計表示。選択ボーン、入力キー数、出力予定キー数、削減率、最大誤差、不連続検出位置、出力後検証の反復概要(§7.3)を表示 |
| `-v, --verbose` | 詳細ログ(通常時は標準出力、`--machine` 併用時は標準エラー) |
| `--quiet` | 進捗のライブ表示を抑制する(警告・統計・終了コードは抑制しない) |
| `--machine` | 出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない(§12) |
| `--describe` | VMD を読まずにオプション定義とプリセット一覧の result イベントを出して終了する。`--machine` を要さず単独で起動でき、入力 positional も要求しない独立メタ操作(§12.3) |
| `--version` | バージョンを表示して終了する |

削減処理中は、規約([CLI インターフェース規約](../../docs/conventions/cli-interface.md) §6.1)が
定める書式で標準エラーへ1行のライブ表示をする。工程名はカメラ・ボーンとも利用者向けの共有工程名
([terminology.md](../../docs/conventions/terminology.md) の「処理工程の利用者向け名称」)「キーフレーム
圧縮」で統一し、対象は行末の補足(note)で示す。カメラは処理済み
フレーム数の割合で完了数/総数を示し、bezier の重い区間でも再帰分割の途中で部分区間が確定する
ごとに進むため表示が長く停滞しない。`reduce_camera_track` が返す補足文字列(出力後検証区間など
重くなりうる段階の注記)があれば「カメラ <補足>」として note に残し、停滞表示でないことを示す。
ボーンは削減対象ボーン(選択され2キー以上を持つトラック)ごとに、着手前の対象ボーン名を note と
して示してから完了数/総数を進める(カメラのような区間単位の細粒度進捗は行わない)。stderr が
リダイレクト・パイプの場合は表示せず、出力・警告・終了コードには影響しない。`--quiet` はこの
ライブ表示だけを抑制し、警告・統計・終了コードは変えない。機械モード(`--machine`。§12)では
ライブ表示を無効化し、同じ進捗を progress イベントとして発行する(端末判定に依存しない)。
`--quiet` が抑制するのはライブ表示だけで、機械モードの progress イベントには影響しない。

`--list-bones` は出力VMDを作らず、ボーン名、ボーンごとのキー数、
include/exclude適用後の選択状態、未一致の選択子警告を表示して終了する。
入力VMDの読み込みに成功すれば終了コード0とする。
`--dry-run` は通常の削減計画に加えて同じ選択状態を表示する。
`--dry-run` は出力VMDを作らず、テキスト要約を標準出力に出して終了する。

---

## 3. VMD入出力

VMDの読み書き・データモデル・正規化は共通ライブラリ vmd に委譲する
(仕様: ../../libs/vmd/vmd-io.md。補間評価: ../../libs/vmd/vmd-interp.md)。
本章はsparsevmdとしての利用要件のみ規定する。

### 3.1 入力
- 対象セクション(camera/bone)は内部作業ビューで正規化する。
  フレーム順にソートし、同一キー重複は後勝ちとする。
- 対象外セクションは原本を保持し、出力時に無加工で透過する。
- VMDヘッダ(`model_name`)は入力の値をそのまま出力に引き継ぐ(変更しない)。
- 対象キーが1件以下のトラックは削減不能としてそのまま保持する。
- 入力がすでに疎なVMDでも処理可能。入力補間曲線を評価したサンプル列を
  ソースモーションとして扱い、再度フィットする。
- カメラとボーンが混在するVMDでは、指定された `--target` のみ再構築する。
- 通常処理では、指定された `--target` が対象とするセクションのキーが
  入力VMDに1件も存在しない場合はエラー(終了コード1)。
  `--target all` でカメラ0件・ボーンありの場合はボーンのみ処理し、
  ボーン0件・カメラありの場合はカメラのみ処理する。
  ボーンセクションにキーは存在するが、ボーン選択ルールの結果が0件になる場合は
  入力不正ではなく引数エラー(終了コード2、§2.2)とする。
  対象トラックは存在するが全トラックが1キー以下の場合は、削減不能として出力を作成し、
  dry-runに「削減対象なし」と記録する。

### 3.2 出力
- 対象セクションは削減後のキー列として再構築する。
  このため対象セクションのバイト一致・元のキー順・元の補間ブロック保持は保証しない。
- ボーン選択(§2.2)で非選択になったボーントラック、および削減不能トラック
  (キー1件以下。§3.1)は、削減せず元のフィールド値と補間曲線をそのまま出力する。
  ただしボーンセクション全体は再構築・再ソートして書き出すため、これら非処理
  トラックも含めセクションのバイト一致は保証しない(値と補間ブロックは保持する)。
- `--range` 外の対象キーは、削減区間に隣接するか否かに関わらず、元のフィールド値と
  補間曲線を逐語保持する(変更不可、§6.3)。範囲外への境界キー追加や範囲外キーの補間曲線
  書き換えは行わない。範囲端に接する到達側曲線のうち書き換えるのは範囲内の範囲開始キーのみ。
- 対象外セクションはバイト単位で保持する。
- 出力キーの再構築・量子化・格納の規約(フレーム昇順・生バイト安定ソート・制御点の `[0,127]`
  量子化・到達側格納・区間長1の線形)は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §9 を正とする。
- カメラキーの perspective フラグの設定方針は §4.2 のとおり vmd-reduce.md §1 を正とする。

---

## 4. 削減対象の単位

### 4.1 トラック
- カメラ: VMD内のカメラキー列全体を1トラックとして扱う。
- ボーン: ボーン名ごとに独立したトラックとして扱う。

### 4.2 チャンネルグループ

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §1 を正とする(本書では重複記述しない)。

### 4.3 共有曲線の制約

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §1 を正とする(本書では重複記述しない)。

---

## 5. フィットアルゴリズム

### 5.1 全体手順

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §4 を正とする(本書では重複記述しない)。

### 5.2 スカラー曲線フィット

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §5.1 を正とする(本書では重複記述しない)。

### 5.3 回転曲線フィット

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §5.2 を正とする(本書では重複記述しない)。

### 5.4 制御点探索

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §5.3 を正とする(本書では重複記述しない)。

### 5.5 分割点の選択

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §6 を正とする(本書では重複記述しない)。

---

## 6. 不連続・境界処理

### 6.1 不連続検出

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §7.1 を正とする(本書では重複記述しない)。

### 6.2 境界の保存

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §7.2 を正とする(本書では重複記述しない)。

### 6.3 範囲端

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §7.3 を正とする(本書では重複記述しない)。

---

## 7. 誤差評価

### 7.1 評価タイミング

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §8.1 を正とする(本書では重複記述しない)。

### 7.2 評価指標

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §8.2 を正とする(本書では重複記述しない)。

### 7.3 出力後検証

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §8.3 を正とする(本書では重複記述しない)。

### 7.4 補間曲線の端点速度平滑化(C1近似)

共有疎化エンジンの仕様は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §8.4 を正とする(本書では重複記述しない)。

---

## 8. パッケージ構成

- VMD読み書き・補間評価・カメラ座標変換・補間曲線フィット・分割戦略・不連続検出は
  共通ライブラリ vmd に委譲する。移設前からの利用者向けに、旧来の公開 import パス
  (`sparsevmd.fit`・`sparsevmd.reduce`・`sparsevmd.cuts`)は後方互換として維持し、
  同名モジュールが vmd の実装を re-export する。
- コアはCLI非依存。Jupyter等からトラック単位で試行できるAPIを提供する。

---

## 9. 終了コード

| コード | 意味 |
|---|---|
| 0 | 正常終了 |
| 1 | 入力ファイル不正(VMDでない / 指定 `--target` の対象セクションにキーが存在しない) |
| 2 | 引数エラー(範囲不正・対象ボーンなし・上書き未許可 など) |
| 3 | VMD書き込み失敗 |
| 4 | `--strict` 指定時に許容誤差を満たせない |
| 130 | 協調的な中断(Ctrl-C 等。全ツール共通予約) |

0〜3 は[CLI インターフェース規約](../../docs/conventions/cli-interface.md)(以下「規約」)§5 の基底と同じ意味、
4 はツール固有(`--strict`)。130 は規約 §5 の全ツール共通予約(中断。§12.5)。想定外の内部エラーは
最も近い基底へ寄せて 1 で終え、機械モードでは error イベントの `code` = `internal_error` で識別できる(§12.4)。

---

## 10. テスト要件

テストフレームワーク・実行方法・テストデータの扱いは [vmd.md](../../libs/vmd/vmd.md) §4 の方針に従う
(pytest、決定論的、外部サービス不要)。

エンジン契約のテスト要件(線形・既知ベジェ・局所極値・カメラ回転共有曲線・ボーン回転・
不連続保存・視野角量子化など)は [vmd-reduce.md](../../libs/vmd/vmd-reduce.md) §10
を正とする。本節は sparsevmd の CLI 挙動に固有のテスト項目のみを列挙する。

1. 対象外セクション透過: モーフ/照明/セルフ影/IKがバイト単位で保持されること
2. 範囲指定: `--range` 外の対象キーが逐語保持(変更不可)され、範囲端の必須キー保持・
   下側継ぎ目の範囲内キー再フィット・レポート出力が仕様通りであること
3. range展開: `:END` / `START:` が対象トラック全体の最小/最大フレームで
   1回だけ展開され、トラックごとの実処理範囲が積集合になること
4. dry-run: 出力VMDを作成せず、キー数・削減率・最大誤差・不連続位置を表示すること
5. 引数エラー: `--camera-fov-tol` 0.5未満が引数エラー(終了コード2)となること
6. 再現性: 同一入力・同一引数で出力がバイナリ一致すること
7. strict: `--strict` 指定時、許容誤差を満たせないケースで終了コード4となること
8. カメラ距離カット: `--cut-threshold-camera POS,ROT,DIST` の DIST で距離ジャンプを
   不連続として検出すること

機械モード(§12)の CLI 固有テスト:

9. 機械モード出力: `--machine` の標準出力が有効な JSON Lines(各行が単一 JSON・UTF-8、行区切りが LF
   のみ)で、result または error のちょうど 1 つで終端すること(通常実行・dry-run・list-bones・
   describe・各エラー経路)。
10. `--describe`: 入力無しで動き、§12.3 の 31 オプションと 3 プリセットを固定形で返すこと。
11. inspect: `--machine --dry-run` が VMD を書かず §12.2 の inspect ペイロード(target・keys・
    frame_range・ranges・camera/bones の誤差とカット)を返し、`--target` の各値と範囲・選択の
    組み合わせで camera/bones の null 有無が正しいこと。
12. list_bones: `--machine --list-bones` が §12.2 の list_bones ペイロードを返し、選択子解決不能でも
    `selector_unmatched`(蓄積分)と `selection_unresolved`(理由)の warning の後に result で終端し
    終了コード 0 になること。
13. 構造化エラー: §12.4 の各経路で、機械モードは所定の `code`/`field`/`exit_code` の error イベント、
    非機械モードは標準エラーへの理由 1 行+同じ終了コードになること(§12.4 の全 `code` を網羅)。
14. 中断: 削減段に `KeyboardInterrupt` を注入し、機械モードで `cancelled`(exit 130)の終端イベント、
    非機械モードで理由 1 行+130 になり、出力ファイルが残らないこと。
15. progress: カメラ段がフレーム進捗と補足(出力後検証)を、ボーン段がトラック進捗とボーン名を、
    機械モードで TTY 非依存に発行すること。
16. 既定挙動の回帰: 非機械・正常系の出力 VMD がバイト一致で不変、終了コード不変、dry-run /
    list-bones の標準出力テキスト不変。`--help`/`--version` が `--machine` 併用でも人間向けテキスト+
    終了コード 0 のままであること。
17. 移植性(規約 §10): 非 ASCII(日本語)の入出力パスで動作し、stderr がロケール符号化で表せない
    文字を含む警告でも異常終了しないこと。
18. 警告透過: デコード不能ボーン名を含む入力で `decode-error` の warning イベントが人間向け経路と同じ
    重複集約で出て、不一致選択子で `selector_unmatched`、範囲外 keep-frame で `keep_frame_ignored` が
    出ること。

---

## 11. 既知の制約・運用上の注意

- VMD補間曲線は任意の密モーションを1区間で表現できない。
  特に高周波ノイズ、細かなモーションキャプチャー揺れ、端点間から外れる回転は
  多くのキーが残る。
- shakevmdの高密度ノイズを強く削減すると、手ぶれの細部は丸くなる。
  忠実度を重視する場合は `precise`、編集しやすさを重視する場合は `balanced` 以上を使う。
- モーションキャプチャーの足滑りや接地破綻は、本ツールでは修正しない。
  sparsevmdは「元モーションの近似」であり「モーション補正」ではない。
- 視野角はVMD上で整数度保存のため、細かな視野角変化は丸め誤差が出る。
- 対象セクションは再構築されるため、入力VMDのキー順や補間ブロックの生バイト保持は
  期待しないこと。原本保持が必要なセクションは `--target` から外す。

### 11.1 性能上の制約と推奨運用

- `bezier` モードは高品質だが `linear` より計算量が大きい。出力後検証(§7.3)で許容超過があれば
  区間を密化して再フィットするため、長尺・高密度入力ではフィット回数が増えて処理時間が伸びる
  (フィットコストの支配項は共有エンジンの正本 [vmd.md](../../libs/vmd/vmd.md) §6)。
- 長尺・高密度入力で待ち時間が問題になる場合の運用回避策:
  - `--range` で区間を分割して必要部分だけ処理する。
  - `--preset aggressive` で許容を緩めてキー数とフィット回数を減らす。
  - `--max-segment-frames` を小さくして1区間あたりの探索を軽くする。
  - 速度最優先なら `--curve-mode linear`(補間曲線フィットを行わない)。
- `shakevmd --smooth` は `curve-mode=bezier` + aggressive 相当 + `max-segment-frames=5` を
  固定設定として使う(短い区間で高速にフィットする)。
- 既定設定で対話的に待てない時間まで未完了になる状態は性能回帰として扱う。
- フィットの性能契約(采否の不変条件・ファストパスの既定と `force_bezier` によるオプトアウト)は、
  共有エンジンの正本 [vmd.md](../../libs/vmd/vmd.md) §6 を参照する(ここに重複して書かない)。

---

## 12. 機械モード(機械可読インターフェース)

機械モードは、他のソフトウェアが sparsevmd を子プロセスとして呼ぶための構造化出力を提供する。
共通契約(イベント種別の語彙・終端規則・チャネル固定・stdout の UTF-8/LF 固定・終了コードの基底)は
規約 §3〜§6・§8・§10 と、共有基盤 [cli_events](../../libs/cli_events/cli_events.md) が正本であり、本節は
sparsevmd 固有のイベントペイロードと `code` 値だけを定める。イベント送出は共有基盤 cli_events に委譲する。

### 12.1 チャネルと終端

- **構造化出力モード**は `--machine` 指定時と `--describe` 指定時(規約 §3。`--describe` は人間向け
  既定を持たない独立メタ操作)。どちらかが argv にあれば引数解析前に先取り判定し、
  使用法エラー・想定外エラーも error イベントで終端する
  (例: `--describe` と未知オプションの併用も `bad_argument` イベント+終了コード 2)。
- 構造化出力モードの標準出力は §12.2 のイベントのみ。バイナリストリームとして
  標準出力へ UTF-8・行区切り LF で書き、ロケール符号化・CRLF 変換に依存しない
  (規約 §10)。
- 人間向け標準エラーは、ロケール符号化で表せない文字を置換して出し、符号化失敗でプロセスを
  落とさない(規約 §10。機械モードに限らず常に適用する)。`--verbose` の詳細ログは機械モードでも
  従来どおり標準エラーへ出す(機械利用側は解釈しない)。
- ストリームは result または error のちょうど 1 つで終端する。終端は CLI 本体の単一経路で送出し、
  終端後の送出は拒否される。
- `--help` / `--version` は `--machine` 併用でも人間向けテキストを出して終了コード 0 で終わり、イベント
  ストリームには載せない(規約 §3 のメタ操作の例外)。
- イベント契約の進化は規約 §4.1 に従う(フィールド・種別・`code` の追加=MINOR、削除・意味変更=MAJOR。
  受信側は未知要素を無視できる前提)。

### 12.2 イベントペイロード

- **progress**: `{type:"progress", stage, done, total, note, elapsed}`。`stage` は安定 id
  `"camera"`(キーフレーム圧縮・カメラ対象)/ `"bone"`(キーフレーム圧縮・ボーン対象)。各段は開始時に
  `done=0, total=null, note:"",
  elapsed:0.0` を 1 本出す。`camera` 段は `reduce_camera_track` の進捗通知(処理済みフレーム数・
  全範囲フレーム総数・補足文字列。出力後検証区間では補足が付く)を `done`/`total`/`note` で出す。
  `bone` 段は削減対象ボーン(選択され 2 キー以上を持つトラック)1 件の完了ごとに `done`/`total`
  (`total`=対象トラック数)、`note`=ボーン名で出す。`elapsed` は段開始からの経過秒(float)。処理しない
  段(`--target bone` のカメラ段など)はイベント自体を出さない。
- **warning**: `{type:"warning", code, message, section}`。`section` は対象セクション名の配列(該当が
  無ければ `null`)。割り当て:
  - `vmd.io.read` の警告は `VmdWarning.code`(ハイフン区切り。現行値 `decode-error` /
    `sections-missing`)をそのまま透過し、`section` は `VmdWarning.section` があれば単一要素配列
    `[section]`、無ければ `null`(`sections-missing` は `section` を持たない)。人間向け経路と
    同じ基準(code・section・message の同一組は 1 件)で重複をまとめる。
  - ボーン選択の不一致警告(不在の exclude 名・1 件も一致しない glob/group。選択解決ハードエラー
    到達前の蓄積分を含む): `selector_unmatched`、`section` は `null`。文言は現行の警告文字列を
    `message` に載せる。
  - `--list-bones` で選択子が解決不能(通常経路なら §12.4 の `bone_selection_invalid`
    で終了コード 2)な場合のその理由: `selection_unresolved`、`section` は `null`、`message` は例外の
    文言。検査モードは現行どおり一覧表示を続けて終了コード 0 なので error ではなく warning とし、
    蓄積分(`selector_unmatched`)の後に 1 件出す。
  - 全削減範囲外の `--keep-frame` の無視: `keep_frame_ignored`、`section` は `null`。
- **result**: 正常終了の終端イベント。`mode` で形が決まる:
  - `mode:"reduce"`(通常実行): `{type:"result", mode:"reduce", output, target, camera, bone, reduced}`。
    `output` は書き出しパス(文字列)。`camera`/`bone` は各セクションを処理対象にしたとき
    `{input_keys, output_keys}`(セクション全キー数。非選択保持ぶんを含む)、対象外なら `null`。
    `reduced` は実際に削減が起きたか(全トラック 1 キー以下・有効範囲空なら `false`)。
  - `mode:"inspect"`(入力検査 `--machine --dry-run`): VMD を書かず
    `{type:"result", mode:"inspect", output:null, target, sections, keys, frame_range, duration_sec,
    ranges, keep_frames, reduced, camera, bones}` を出す。`sections` は入力に存在するセクション名の
    配列、`keys` は `{camera, bone}`(入力キー数)、`frame_range` は処理対象セクションの全キーの
    `[最小フレーム, 最大フレーム]`(対象キーが無ければ `null`)、`duration_sec` は最大フレーム÷30
    (同 `null`)、`ranges` は展開後のグローバル削減範囲の `[start, end]` 配列、`keep_frames` は正規化後の
    指定フレーム配列。`camera` はカメラを処理したとき `{input_keys, output_keys, errors, cuts}`
    (`errors` は軸別最大再生誤差 `{pos_x, pos_y, pos_z, rot_deg, distance, fov}`、`cuts` は不連続検出
    フレームの配列)、それ以外 `null`。`bones` はボーンを処理したとき初出順の
    `{name, selected, input_keys, output_keys, errors, cuts}` の配列(非選択・削減不能トラックは
    `errors`/`cuts` を `null`)、それ以外 `null`。継ぎ目書き換え・分割理由・出力後検証の反復詳細は
    人間向け dry-run / `--verbose` に残し、inspect には載せない(必要になれば規約 §4.1 の後方互換追加で
    拡張する)。
  - `mode:"list_bones"`(`--machine --list-bones`): `{type:"result", mode:"list_bones", bones}`。
    `bones` は初出順の `{name, keys, selected}` の配列。VMD は書かない。選択子が解決不能な場合も
    現行どおり一覧(全件非選択)で終了コード 0 とし、`selector_unmatched`(蓄積分)と
    `selection_unresolved`(理由)の warning イベントを result より先に出す。非機械の `--list-bones`
    は従来どおり人間向けテキスト。
  - `mode:"describe"`(自己記述 `--describe`): `{type:"result", mode:"describe", options, presets}`
    (§12.3)。VMD を読まないので他 mode のキーは載せない。
- **error**: `{type:"error", code, exit_code, field, path, message}`。`field`/`path` は対象が無ければ
  `null`。失敗の終端イベント(§12.4)。

### 12.3 `--describe` の中身

`options` は処理を駆動する引数の配列(メタ/モード操作 `--describe`/`--version`/`--help`/`--machine` は
含めない)。各要素は `{name, type, constraint, default, help, repeat}`(キーは常に 6 つ、該当しない値は
`null`)。`repeat` は同名オプションを複数回指定できるか(bool)。`type` は固定語彙
`"float"`/`"int"`/`"str"`/`"flag"`/`"enum"`/`"compound"`。数値の `constraint` は
`{min, max, exclusive_min}` の 3 キー常設(上限が無ければ `max:null`)、`enum` は `{choices:[...]}`、
`compound` は `{format, fields}`(`fields` の各要素は `{name, type, min, max, exclusive_min}`)、
`flag`/`str` は `null`。`help` は §2 のヘルプ文言。

- 真偽フラグの対は**肯定形の長形式 1 要素だけ**を載せる(型 `flag`。`--cut-detect`/`--no-cut-detect`
  の対は `--cut-detect` の 1 要素)。無効化の起動形は名前に `--no-` を前置した否定形とする
  (規約 §6 の `--x/--no-x` 様式。呼び出し側は `default` が `true` のフラグを無効化するとき否定形を
  発行する)。否定形を別要素として重複列挙しない。
- `--bone-group`/`--exclude-bone-group` の `enum` は自己記述上の値域であり、**argparse の `choices`
  へは追加しない**。未知グループの検証は現行どおり選択解決時のハードエラー(§12.4 の
  `bone_selection_invalid`)であり、§2.2 の検証位置・順序を変えない。

全 31 要素を確定する:

| name | type | constraint | default | repeat |
|---|---|---|---|---|
| `input` | str | null | null | false |
| `--output` | str | null | null(既定は入力名由来 `<入力名>_sparse.vmd` の算出値。規則は help に記す) | false |
| `--overwrite` | flag | null | false | false |
| `--target` | enum | `{choices:["camera","bone","all"]}` | `"all"` | false |
| `--bone` | str | null | null | true |
| `--bone-glob` | str | null | null | true |
| `--bone-group` | enum | `{choices:["core","arms","legs","fingers","ik","mocap"]}` | null | true |
| `--bone-file` | str | null | null | false |
| `--exclude-bone` | str | null | null | true |
| `--exclude-bone-glob` | str | null | null | true |
| `--exclude-bone-group` | enum | `{choices:["core","arms","legs","fingers","ik","mocap"]}` | null | true |
| `--list-bones` | flag | null | false | false |
| `--range` | compound | `{format:"START:END", fields:[START int min:0, END int min:0]}`(各辺省略可。順序制約はトークン横断のため fields に含めず、検証失敗は `bad_argument`/`range_invalid` で返す) | null | true |
| `--preset` | enum | `{choices:["precise","balanced","aggressive"]}` | `"balanced"` | false |
| `--bone-pos-tol` | float | `{min:0, max:null, exclusive_min:false}` | null(プリセット値) | false |
| `--bone-rot-tol` | float | `{min:0, max:null, exclusive_min:false}` | null(プリセット値) | false |
| `--camera-pos-tol` | float | `{min:0, max:null, exclusive_min:false}` | null(プリセット値) | false |
| `--camera-rot-tol` | float | `{min:0, max:null, exclusive_min:false}` | null(プリセット値) | false |
| `--camera-distance-tol` | float | `{min:0, max:null, exclusive_min:false}` | null(プリセット値) | false |
| `--camera-fov-tol` | float | `{min:0.5, max:null, exclusive_min:false}` | null(プリセット値) | false |
| `--max-segment-frames` | int | `{min:1, max:null, exclusive_min:false}` | null(無制限) | false |
| `--min-segment-frames` | int | `{min:1, max:null, exclusive_min:false}` | 1 | false |
| `--curve-mode` | enum | `{choices:["bezier","linear"]}` | `"bezier"` | false |
| `--strict` | flag | null | false | false |
| `--cut-threshold-camera` | compound | `{format:"POS,ROT,DIST", fields:[POS float min:0, ROT float min:0, DIST float min:0]}` | `[5.0, 20.0, 5.0]` | false |
| `--cut-threshold-bone` | compound | `{format:"POS,ROT", fields:[POS float min:0, ROT float min:0]}` | `[1.0, 30.0]` | false |
| `--cut-detect` | flag | null | true | false |
| `--keep-frame` | int | `{min:0, max:null, exclusive_min:false}` | null | true |
| `--dry-run` | flag | null | false | false |
| `--verbose` | flag | null | false | false |
| `--quiet` | flag | null | false | false |

compound の数値 fields は `{name, type, min, max, exclusive_min}` で、省略した `max` は `null`、
`exclusive_min` は `false`。

`presets` は各要素 `{name, values}` の配列。`name` は品質プリセット名(`precise`/`balanced`/
`aggressive`)、`values` は `{bone_pos_tol, bone_rot_tol, camera_pos_tol, camera_rot_tol,
camera_distance_tol, camera_fov_tol}`(§2.4 の表の値)。

### 12.4 構造化エラー

失敗は終了コードに加え、構造化出力モード(`--machine`・`--describe`。§12.1)では error イベントで
「どのフィールド/パスが・なぜ」を返す。それ以外では理由を標準エラーへ最低 1 行出す(書式
`error: <message>`。トレースバックは出さない)。検証の位置・順序は現行のまま(§2.2)。

| 事象 | `code` | `field` | `exit_code` |
|---|---|---|---|
| argparse 検出(未知オプション・型エラー・choices 外・positional 欠落・`--range`/`--keep-frame`/カット閾値の書式不正)、および解析後の単一オプション検証(`--min-segment-frames` < 1・`--max-segment-frames` < 1) | `bad_argument` | argparse が示す引数名、解析後検証は該当オプション名 | 2 |
| `--min-segment-frames` > `--max-segment-frames` | `segment_bounds_conflict` | `null`(2 オプションにまたがる) | 2 |
| 許容誤差の検証失敗(負値・非有限・fov < 0.5) | `bad_tolerance` | `null`(起因フィールド名は例外文言として `message` に載る) | 2 |
| 入力パスが不在・通常ファイルでない | `input_not_file` | `"input"` | 2 |
| `--bone-file` パスが不在・通常ファイルでない | `bone_file_not_file` | `"--bone-file"` | 2 |
| `--bone-file` の読み込み・解析失敗(UTF-8 デコード不能等) | `bad_bone_file` | `"--bone-file"` | 2 |
| `--target camera` とボーン選択の同時指定 | `target_selection_conflict` | `null` | 2 |
| 出力先に既存ファイルがある・`--overwrite` 未指定 | `output_exists` | `"--output"` | 2 |
| ボーン選択のハードエラー(空文字名・include/exclude 重複名・明示 `--bone` 名の不在・選択結果 0 件) | `bone_selection_invalid` | `null`(対象名は `message` に載る) | 2 |
| `--range` の展開・正規化失敗(重複・省略端解決後の逆順) | `range_invalid` | `"--range"` | 2 |
| 入力が VMD でない・破損 | `not_vmd` | `"input"` | 1 |
| 指定 `--target` の対象セクションにキーが無い | `no_target_keys` | `"input"` | 1 |
| `--strict` で許容誤差を満たせない | `strict_tolerance_unmet` | `null` | 4 |
| 出力書き込み失敗 | `write_failed` | `"--output"`(+ `path`) | 3 |
| 上記いずれにも当たらない想定外の内部エラー | `internal_error` | `null` | 1 |
| 協調的な中断(Ctrl-C 等) | `cancelled` | `null` | 130 |

- `not_vmd` は現在握り潰している例外の種別・文言を `message` に載せる。`bone_selection_invalid` の
  送出前に、蓄積済みの不一致警告を warning イベントとして先に出す
  (人間向け経路の順序と同じ)。
- 構造化出力モードの argparse エラーは `bad_argument` イベントへ振り替える。`field` の抽出規則は
  [cli_events.md](../../libs/cli_events/cli_events.md) §4 が正。
- `internal_error` は引数解析後の本体をトップレベルで捕捉して畳む。`KeyboardInterrupt` は内部エラーで
  なく中断(`cancelled`/130)として手前で分岐する(§12.5)。

### 12.5 中断と出力の原子性

- VMD 出力は一時ファイル+原子置換で行う。書き込みは全計算後に 1 回だけ
  起きるため、途中終了で中途半端な出力ファイルは残らない。
- CLI は引数解析後の本体で `KeyboardInterrupt` を捕捉し、構造化出力モードでは `cancelled` の
  error イベントでストリームを終端、それ以外では理由を標準エラーへ 1 行出し、どちらも終了コード 130
  で終える。POSIX シグナル API には依存せず、`KeyboardInterrupt`(Ctrl-C)の捕捉で畳む(sparsevmd の
  削減は単一プロセスで走り、子プロセスは持たない)。
- 進捗のライブ表示は中断・例外経路でも行を閉じてから終える(try/finally で保証する)。
  機械モードではライブ表示自体を無効化する。
- 呼び出し側がプロセスを強制終了した場合は終端イベントを出せないまま途切れる(規約 §4 の終端保証の
  唯一の例外。§8)。出力の原子性により中途半端な出力ファイルは残らない。
