# 修正計画書: sparsevmd リデュース機能の mmd_toolbox 切り出しと shakevmd からの直接利用

ステータス: 計画(未着手)。対象ブランチ: `develop`(sparsevmd マージ済み)。

関連: [design-60fps-smooth.md](design-60fps-smooth.md)(設計判断)、
[../sparsevmd/performance-fix-plan.md](../sparsevmd/performance-fix-plan.md)(性能改修)。

---

## 1. 背景と目的

- **60fps問題**: shakevmd の出力は全整数フレームに密キー + 線形補間。60fps/無制限再生で各フレームの
  速度不連続(角)がブルブルとして見える。
- **検証結果**: 密出力をベジェ補間へ置換すると問題は解消する(MMD 目視で確認)。検証は
  「shakevmd で密ベイク → sparsevmd の reducer(`--curve-mode bezier --preset aggressive
  --max-segment-frames 5`)で疎ベジェ化」という**2プロセス・中間ファイル経由のパイプライン**で行った。
- **目的**: このパイプラインを廃し、sparsevmd のリデュース機能を共通ライブラリ `mmd_toolbox` へ
  切り出して、**shakevmd がプロセス内で直接呼ぶ**。1ツール・1実行で滑らかな出力を得る。
- **採用設定(検証で確定)**: `curve-mode=bezier` / `preset=aggressive 相当の許容` /
  `max-segment-frames=5`。

## 2. 方針

- 検証で実際に動いた「**sparsevmd の reducer を密出力へ適用**」する構成をそのまま製品化する
  (design メモの B-1 を、CLI パイプラインでなく**ライブラリ直接利用**で実現)。
- shakevmd が自前の解析ノイズを極値区間で直接フィットする案(B-2')はさらに忠実だが、本計画の
  **スコープ外**(将来の精度・性能改善余地として §9 に記載)。
- **レイヤリング(CLAUDE.md)**: 「読み書き・MMD互換評価はツールでなく mmd_toolbox」。ベジェ
  **フィット**は既存の補間**評価**(`mmd_toolbox.vmd.interp`)の対であり、フォーマット層として
  mmd_toolbox に置くのが妥当。ツール → ツール依存(shakevmd → sparsevmd)は作らない。

## 3. 抽出境界

### 3.1 mmd_toolbox へ移す(共通リデュースエンジン)

| 現在地 | 対象 |
|---|---|
| `sparsevmd/fit.py` | **モジュール全体**。原始関数(`fit_bezier_curve` / `_fit_coeff_curve` / `_bezier_y_at` / `_bez` / `_quantize_cp` / `_clip01` / `_round_half_up` / `_BEZIER_INITS` / `_BEZIER_LINEAR_CP`)+ チャンネル評価器(`LinearScalarChannel` / `EuclideanVectorChannel` / `FovChannel` / `CameraRotationChannel` / `BoneRotationChannel`)+ **内部ヘルパ `_normalize` / `_select_worst` / `_axis_curve` / `_bezier_axis_pred`** + **quaternion ユーティリティ `_quat_dot` / `_quat_normalize` / `_quat_conj` / `_quat_mul` / `_quat_slerp` / `_quat_angle_deg`**(reduce.py は `_quat_angle_deg` と `_round_half_up` を fit.py から import するため、これらの公開先も揃える) |
| `sparsevmd/cuts.py` | `detect_cuts_camera` / `detect_cuts_bone` / `assemble_boundaries` / `perspective_cut_frames` / 角度ヘルパ |
| `sparsevmd/reduce.py` | **モジュールほぼ全体**。汎用分割(`reduce_track` / `_process_segment` / `_worst_channel` / `_presplit` / `StrictError`)+ camera 経路(`reduce_camera_track` / `build_camera_keys` / `camera_interp_bytes` / 境界ヘルパ)+ bone 経路(`reduce_bone_track` / `build_bone_keys` / `bone_interp_bytes`)※ + **出力後検証(§7.3)`verify_camera_track` / `verify_bone_track`**(`reduce_*_track` が内部で呼ぶ)+ **継ぎ目(§6.3)`_camera_seam_interp` / `_bone_seam_interp` / `_nearest_source_before` / `_interval_clear_of_ranges`** + **誤差測定 `measure_camera_errors` / `measure_bone_errors`**(レポート用)。`reduce_track` の `splits`/`progress`、`diagnostics` 引数も含めて移送 |
| `sparsevmd/presets.py` | `Tolerances` dataclass(許容誤差の器。**preset 名 → 値の表は sparsevmd に残す**、後述) |
| `sparsevmd/sample.py` | `perspective_series`(camera 経路が使う小ヘルパ) |

※ bone 経路は shakevmd は使わないが、汎用分割を共有するため一緒に移すとレイヤが綺麗。
camera のみ移して bone は sparsevmd 残置でも可(Step 1 で確定)。

配置案: `mmd_toolbox/vmd/fit.py`(フィット)、`mmd_toolbox/vmd/reduce.py`(分割+トラック構築)、
`cuts` は reduce へ同梱 or `mmd_toolbox/vmd/cuts.py`。

### 3.2 sparsevmd に残す(ツール固有)

- `cli.py`、`presets.py` の **preset 名 → tol 表**(品質プリセットは sparsevmd の運用ポリシー)、
  `selection.py`、`ranges.py`、`report.py`。
- `report.py` は誤差測定 `measure_camera_errors` / `measure_bone_errors` を **mmd_toolbox から import** して使う
  (関数本体は §3.1 で mmd_toolbox へ移送)。
- sparsevmd の reducer 関数群(`reduce_camera_track` 等)は mmd_toolbox の同名関数を呼ぶ**薄いラッパ**に縮む
  (または直接 re-export)。**挙動は不変**で、`pytest sparsevmd` が抽出前後で同一であること。

### 3.3 shakevmd の利用

- `bake()` で密キーを生成した後、**プロセス内で**、生成した `BakeResult.camera_keys`(メモリ上の
  `CameraKey` 列)を **そのまま** mmd_toolbox の `reduce_camera_track(source_keys=...)` へ渡して
  疎ベジェキーへ変換し、**最終の滑らかVMDだけ**を書き出す。
- **必須要件: 中間VMDを生成しない**。密キーをディスクへ書いてから再読込する構成は採らない。
  `reduce_camera_track` は既に `source_keys`(キー列)を引数に取り、ファイルI/Oに依存しないので、
  bake → reduce は純粋なメモリ間ハンドオフで実現できる(`io.write_file` は最終出力1回のみ)。
- 許容は固定値(検証で選んだ aggressive: `camera_pos=0.10` / `camera_rot=0.25` /
  `camera_distance=0.10` / `camera_fov=1.00`)を **shakevmd 側の定数**として持つ。sparsevmd の
  preset 表には依存しない(ツール間依存回避)。
- **呼び出し時の全引数**(`reduce_camera_track` の現行シグネチャに合わせる):
  `source_keys`(bake 出力キー列)、`ranges`(位置引数。bake と同じ範囲。全範囲なら `[(first,last)]`)、
  `tols`(上記固定 `Tolerances`)、キーワード必須 `cut_thresholds`(`(POS,ROT,DIST)`)、
  `keep_frames=()`、`no_cut_detect`(下記参照)、`min_seg=1`、`max_seg=5`、`strict=False`、
  `curve_mode="bezier"`。オプション `diagnostics`/`progress` は任意。
- **カット整合**: shakevmd はベイク時に独自のカット検出(`cut_pos_threshold` / `cut_rot_threshold`)を
  行う。reduce 側のカット検出と二重・矛盾しないよう、`cut_thresholds` をベイク時と整合させるか、
  ベイクで境界処理済みなら `no_cut_detect=True` にするかを Step 3 で確定する(distance 閾値の扱いも含む)。
- **`Tolerances` の bone 項目**: 現行 `Tolerances` は 6 フィールド(`bone_pos` / `bone_rot` / camera 4 項目)
  すべて必須で既定値が無い。shakevmd は camera 4 項目に加え `bone_pos` / `bone_rot` にも有効なダミー値
  (camera 経路では不参照)を入れた `Tolerances` を渡す。あるいは `Tolerances` に既定値を付与するか
  camera 専用許容型を切る。実装時に確定(Step 1/3)。
- **resolved ranges の共有**: `bake()` は渡された `ranges` を `_snap()` で既存キーへスナップ・正規化した
  `resolved` を内部で使うが、これはローカル変数で `BakeResult`(`camera_keys` / `warnings` のみ)に含まれない。
  CLI パイプラインとの同値性のため、`reduce_camera_track` の `ranges` には**この `resolved` を渡す**必要がある。
  → Step 3 で `BakeResult` に `resolved`(スナップ後の範囲)を持たせ、それを reduce に渡す
  (全範囲ベイクなら `[(first,last)]` で代替可)。

## 4. 性能(クリティカルパス)

- 直接利用は**ベイクのたびにリデュースが走る**。現状の reducer は 4541 キー / 7440 フレームで
  `max5` 約10分・`max10` 15分45秒と**非実用**。
- したがって [performance-fix-plan.md](../sparsevmd/performance-fix-plan.md) の改修
  (区間フィットのメモ化・**線形ファストパス**・`_bezier_y_at` のベクトル化)を**本計画に取り込む**。
  性能改修は「目的(実用的な直接利用)に必要な手段」であり、本計画の**必須フェーズ**(§5 Step 2)。
- 追加の高速化レバー: `max-segment-frames` を小さくすると総時間はほぼ比例して減る(超線形の
  区間内フィット費用を抑えるため)。採用値 `max5` は速度面でも有利。

## 5. 実装ステップ

### Step 0: 前提確認
- `develop` の `bake.py` の現行シグネチャ・既定値(`motion_damp`/`settle` 等チューニング反映状態)を確認。
- `pytest` 基準を記録(既知の3失敗=`_quat_angle_deg` の acos 精度は別件として切り分け)。

### Step 1: リデュースエンジンの抽出(挙動不変リファクタ)
- §3.1 を mmd_toolbox へ移動。sparsevmd は再 import。公開 API シグネチャは維持。
- 完了条件: `pytest sparsevmd` が抽出前と同一結果。`mmd_toolbox` 側に移したコードのテストも移送/新設。

### Step 2: 性能改修(直接利用を実用化)
- performance-fix-plan.md の Step1(メモ化)+ 線形ファストパス + 必要なら Step2(ベクトル化)。
- 完了条件: `max5` 全長が実用時間(目標: 数十秒〜数分)で完了。出力は改修前と許容内一致。

### Step 3: shakevmd からの直接利用
- `bake()` 後段に reduce 呼び出しを追加。出力を疎ベジェキーにする経路を新設。
- 露出方法: まず **opt-in フラグ**(例 `--smooth`)で追加。既定は当面据え置き(§6 で既定化を判断)。
- 固定設定(§3.3)を内部定数化。

### Step 4: FOV 温存との整合
- 現行 shakevmd は緩いズームの FOV 区間を温存(密ベイクしない)。reducer の FOV チャンネル
  (整数度ベジェ)がこの特例を**置換できるか**を検証し、二重処理・矛盾を除く。

### Step 5: 仕様・テスト・パイプライン撤去
- `shakevmd.md` に滑らか出力経路を記載。検証用の中間ファイル/別プロセス手順を撤去 or 置換。
- shakevmd 直接出力が sparsevmd CLI 経由(同設定)と整合することをテストで担保(基準は §6 の (a)/(b) に従う。
  バイト完全一致は前提にしない)。

### Step 6: 既定化の判断
- opt-in での評価後、ベジェ出力を**既定**にするか判断(出力フォーマット変更=後方互換に注意)。

## 6. テスト・検証

- **抽出の挙動不変**: `pytest`(特に `sparsevmd`)が抽出前後で同一。
- **同値性の基準(exact か f32量子化後か を明記)**: 直接パスは **in-memory の f64 ソース**を reduce し、
  旧 CLI パイプラインは**中間VMDの f32 往復後のソース**を reduce する(`io` はカメラ値を f32 で読み書き)。
  reduction 判定の入力精度が違うため、**バイト完全一致は期待しない**。基準は次のどちらかを採用し明記する:
  - (a) **本番採用**: 直接出力が**同じ aggressive 許容内**で元動作を再現すること(f64 直接の方がむしろ忠実)。
  - (b) **バイト級の回帰テストが要る場合のみ**: ソースを先に f32 往復(VMD write/read)してから reduce し、
    CLI パスと一致させる。
  production は (a)、CLI パスとの厳密一致を確認したいテストでのみ (b) を使う。
- **知覚再確認**: 代表素材を 60fps 目視(回帰確認)。
- **性能ベンチ**: 改修前後で `max5` 全長の時間・キー数・最大誤差を比較。

## 7. リスクと対策

| リスク | 対策 |
|---|---|
| 直接利用で baking が遅い(最大リスク) | Step 2 を必須化。`max5` + メモ化 + 線形ファストパスで実用時間へ |
| FOV 温存と reducer の二重処理 | Step 4 で特例を reducer 側へ一本化 |
| ブランチ分岐(bake.py のチューニング状態) | Step 0 で現行 `develop` の実態を確認してから着手 |
| 既存3テスト失敗(acos 精度) | 別件として切り分け(本計画の完了条件に含めない/別途修正) |
| 既定化による後方互換 | Step 6 まで opt-in。既定化は別判断 |

## 8. 作業順序

1. Step 0 前提確認・ベンチ基準記録。
2. Step 1 抽出(挙動不変)→ `pytest` 同一確認。
3. Step 2 性能改修 → ベンチで実用時間確認。
4. Step 3 shakevmd 直接利用(opt-in)。
5. Step 4 FOV 温存整合。
6. Step 5 仕様・テスト・パイプライン撤去。
7. Step 6 既定化の判断。

## 9. スコープ外

- **B-2'**(shakevmd が自前の解析ノイズを極値区間で直接フィット): さらに忠実・高速化余地。
  本計画(reducer のライブラリ化+直接利用)の後の精度改善候補。
- shakevmd からの bone 経路利用(shakevmd はカメラのみ)。
- sparsevmd の品質プリセット体系の変更。
