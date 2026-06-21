# sparsevmd 性能問題修正計画

ステータス: 現役(残作業あり)。

本計画は、実装済みの `sparsevmd` / `shakevmd --smooth` を前提に、長尺・高密度カメラ入力で
残っている性能リスクを潰すための作業だけを扱う。抽出済みコードの所在は `mmd_toolbox/vmd/fit.py` と
`mmd_toolbox/vmd/reduce.py`、CLI 側の利用箇所は `sparsevmd/cli.py` と `shakevmd/cli.py`。

## 1. 目的と要件

前提:

- `shakevmd` の密ベイク出力は(FOV 温存区間=ゆっくりズームを除く)整数フレームにキーを持ち、線形補間の速度不連続が 60fps / 無制限再生で微振動として見える。
- 密ベイク結果を `bezier` 曲線で疎化すると微振動が目視で改善することは検証済み。
- `shakevmd --smooth` は既定 on で、この滑らか出力経路自体は実装済み。
- `shakevmd --smooth` の固定設定は `curve-mode=bezier`、aggressive 相当の許容、`max-segment-frames=5`。
- この性能計画の対象は、解決済みの滑らか出力経路を長尺・高密度入力でも待てる時間で動かすこと。

目的:

- `sparsevmd` の `bezier` モードを、長尺・高密度カメラ入力でも実用範囲の処理時間に収める。
- `shakevmd --smooth` の既定経路で、代表素材を実用時間内に処理できる状態を維持する。

要件:

- `bezier` 出力品質を壊さない。ベクトル化などを入れる場合も、出力キー数と最大誤差を比較して許容内に収める。
- `shakevmd --smooth` 用の aggressive 相当 + `max-segment-frames=5` 経路を性能目標に含める。
- `balanced/bezier` 全長と `aggressive/max5` 全長の実測を残し、遅い場合は追加最適化の判断材料を残す。
- 出力後検証ループの反復状況を diagnostics/report で追えるようにする。
- 性能上の制約と推奨運用を仕様書または本計画書に残す。

## 2. 未実装 / 要確認

- `_bezier_y_at()` はまだ単点単位の純 Python ループで、ベクトル化されていない。
- `least_squares()` に `max_nfev` は設定していない。
- 出力後検証ループの反復回数・追加フレーム数は diagnostics にまだ出していない。
- 既定 `balanced` / `bezier` の `tuning/dense_camera.vmd` 全長性能は未測定。
- `shakevmd --smooth` 用の aggressive/max5 経路は、短・中レンジでは実用時間に改善している。
  代表長尺素材での全長ベンチと目視回帰は未記録のため、追加計測対象に残す。

## 3. 実測(現状ベースライン)

環境: Windows / Python 仮想環境 `.venv` / `tuning/dense_camera.vmd`。
計測は `--dry-run` で出力ファイルを生成せずに実施する。
ベンチ入力 `tuning/dense_camera.vmd` と `workspace/` の代表素材は `.git/info/exclude` でローカル除外されており
リポジトリに追跡されない。したがって下表の秒数は特定環境・特定素材での**過去測定の参考値**であり、絶対基準には
しない。**回帰判定**は、着手時に同一セッション・同一環境・同一入力で変更前後を計測した**相対差**で行う(Step A で
ベースラインを取り直す)。**実用時間の完了判定**は、Step A で素材ごとに設定する受入上限(§8)と相対差を併用する。

```powershell
$t=Measure-Command { .\.venv\Scripts\sparsevmd.exe tuning/dense_camera.vmd --dry-run --range 0:60 *> $null }
$t=Measure-Command { .\.venv\Scripts\sparsevmd.exe tuning/dense_camera.vmd --dry-run --range 0:180 *> $null }
$t=Measure-Command { .\.venv\Scripts\sparsevmd.exe tuning/dense_camera.vmd --dry-run --range 0:720 --preset aggressive --max-segment-frames 5 *> $null }
```

| 設定 | range | 実測 |
|---|---:|---:|
| default (`balanced`, `max-segment-frames=180`) | `0:60` | 5.21 s |
| default (`balanced`, `max-segment-frames=180`) | `0:180` | 18.65 s |
| `--preset aggressive --max-segment-frames 5` | `0:720` | 13.92 s |

暫定判断:

- 実装済みのメモ化と早期終了の効果は短・中レンジの実測に出ている。
- `shakevmd --smooth` が使う aggressive/max5 経路は、短・中レンジでは実用的な速度になっている。
- 既定 `balanced` の `0:60` / `0:180` は改善済みだが、`tuning/dense_camera.vmd` 全長が数分以内に収まるかは未測定。

## 4. 残る支配要因

現状の `mmd_toolbox/vmd/fit.py` / `mmd_toolbox/vmd/reduce.py` で、まだ重い可能性が高い箇所:

1. `mmd_toolbox/vmd/fit.py` の `_bezier_y_at()`。
   - `least_squares()` の残差評価ごとに内部点を Python ループで評価する。
   - Newton 法 + 二分探索フォールバックも点ごとに走る。
2. 回転系チャンネルの `_fit_coeff_curve()`。
   - `resid_at(lambda x: _bezier_y_at(...))` が内部点ごとに Python 関数呼び出しを重ねる。
3. 出力後検証ループ。
   - 許容超過フレームを追加するたびに `build_*_keys()` と `verify_*_track()` を繰り返す。
   - 現状は発生回数を diagnostics に残していないため、悪化箇所を後から追いにくい。

## 5. 修正方針

残作業は、品質に影響しにくい順に進める。

1. 全長ベンチと diagnostics の見える化で、現状の残ボトルネックを再確認する。
2. 出力後検証ループの反復回数・追加フレーム数を diagnostics/report に出す(Step C)。
3. `_bezier_y_at()` のベクトル化で残差評価の Python ループを削る(Step B)。
4. それでも既定 `balanced` が遅い場合に限り、`least_squares(max_nfev=...)` と初期値試行数の制御を検討する(Step D)。

`max_nfev` や初期値削減は、制御点品質・キー数に影響する可能性があるため後回しにする。

## 6. 実装ステップ

### Step A: ベンチ基準を固定する

このステップで取る計測値が、回帰・品質比較のベースライン(変更前)になる。変更後に同じ手順・同じ環境・
同じ入力で再計測し、相対差で判定する(§7)。ベンチ入力は追跡外(§3)なので、手元の代表素材で前後比較する。

対象データ:

- `tuning/dense_camera.vmd`
- 追加計測対象: `workspace/` の実利用カメラ VMD 代表素材

計測:

```powershell
pytest sparsevmd

$cases = @(
  @{ name='balanced 0:60'; args=@('tuning/dense_camera.vmd','--dry-run','--range','0:60') },
  @{ name='balanced 0:180'; args=@('tuning/dense_camera.vmd','--dry-run','--range','0:180') },
  @{ name='balanced full'; args=@('tuning/dense_camera.vmd','--dry-run') },
  @{ name='aggressive max5 0:720'; args=@('tuning/dense_camera.vmd','--dry-run','--range','0:720','--preset','aggressive','--max-segment-frames','5') },
  @{ name='aggressive max5 full'; args=@('tuning/dense_camera.vmd','--dry-run','--preset','aggressive','--max-segment-frames','5') },
  @{ name='linear full'; args=@('tuning/dense_camera.vmd','--dry-run','--curve-mode','linear') }
)
foreach ($case in $cases) {
  $t = Measure-Command { .\.venv\Scripts\sparsevmd.exe @($case.args) *> $null }
  Write-Host "$($case.name): $([math]::Round($t.TotalSeconds, 2)) s"
}
```

記録するもの:

- 実行時間。
- 出力キー数(`--dry-run -v` または report)。
- `--report-json` の最大誤差。
- diagnostics の splits 数・cuts 数・seam rewrites 数。
- 出力後検証の反復回数(未実装なら Step C で追加後に記録)。

時間は上記 Measure-Command(`*> $null`)で測る。品質項目(キー数・最大誤差・diagnostics)は時間計測では
取れないので、同じ各ケースを `--report-json <path>` と `-v` を付けて別途1回ずつ実行して取得・記録する。

`shakevmd --smooth` の経路は `--dry-run` では測れない(smooth 削減の前に終了する。`shakevmd/cli.py`)。
代表素材で実出力を書く実行の時間を測る。`shakevmd --smooth` は `reduce_camera_track`(sparsevmd の
aggressive/max5 と同エンジン)を通るため、sparsevmd aggressive/max5 ベンチが主指標で、shakevmd 実出力計測は
その代表素材での確認を担う。これも変更前後を同手順で計測してベースラインにする。

あわせて、素材ごとに「対話的に待てる」受入上限(秒数。目安 数分)を着手者が決めて記録する。これが §8 の実用時間の
合格基準(絶対上限)になり、相対悪化なしと併用する。一律の固定値は追跡外・環境依存のため計画書には書かない。

### Step B: `_bezier_y_at` をベクトル化する

対象:

- `mmd_toolbox/vmd/fit.py`
- ベクトル化の差分テスト追加先: `mmd_toolbox/tests/vmd/` 相当

内容:

- `_bezier_y_at_many(px1, py1, px2, py2, xs)` を追加する。
- `xs` は `numpy.ndarray` として扱う。
- Newton 反復を配列演算で一括実行する。
- 収束しない要素だけ mask して二分探索フォールバックする。
- `fit_bezier_curve()` の `residual()` は list comprehension ではなく `_bezier_y_at_many()` を使う。
- 回転系 `_fit_coeff_curve()` の `residual()`(`resid_at(lambda x: _bezier_y_at(...))`)も同様にベクトル化する。
  チャンネル側の `resid_at` が曲線関数を内部点配列で一括評価できる形にし、§4 の支配要因2 を解消する。
- これに伴い `_fit_coeff_curve()` の量子化後評価 `quantized_err()`(`resid_at(lambda x: interp._solve_factor(...))`)も
  同じ配列契約になる。`interp._solve_factor()` を内部点配列で評価できる経路(ベクトル版または配列ループ)を用意する。
- 単点用 `_bezier_y_at()` はテスト・読みやすさ・既存 import 互換のため残す。

注意:

- `interp._solve_factor()` と完全一致は不要だが、量子化後の採否結果が不安定にならない精度にする。
- `x=0` / `x=1`、線形、ease-in/out、`x1=0`、`x2=1` に近い制御点を比較テストに入れる。

### Step C: 出力後検証ループを diagnostics に出す

対象:

- `mmd_toolbox/vmd/reduce.py`
- `sparsevmd/report.py`
- `sparsevmd/cli.py` の診断表示

内容:

- `verify_camera_track` / `verify_bone_track` の再構築ループについて、範囲ごとに以下を記録する。
  - `verify_iterations`
  - 各反復の `bad_count`
  - 各反復の `added_count`
  - 最終的に追加したフレーム数
- JSON report と verbose diagnostics の両方で追えるようにする。
- 非strictで `added_count` が大きい範囲は、将来の密化早期切替候補として観測できるようにする。

### Step D: 必要な場合だけ最適化回数を制御する

対象:

- `mmd_toolbox/vmd/fit.py`
- 仕様記載が必要になった場合の更新先: `sparsevmd/sparsevmd.md`

検討内容:

- `least_squares(..., max_nfev=N)` を内部定数で導入する。
- まず線形初期値を試し、量子化後の最終誤差が許容内なら追加初期値を省略する。
- 実装済みの `early_exit_err` で全長ベンチ目標を満たす場合は実装しない。

注意:

- 既知ベジェ復元テストが壊れる可能性がある。
- キー数増加と最大誤差を実データで比較してから採用する。

### Step E: 仕様書の更新

対象:

- `sparsevmd/sparsevmd.md`
- `shakevmd --smooth` の固定設定や運用注意を変更する場合の更新先: `shakevmd/shakevmd.md`

追記・確認する内容:

- `bezier` モードは高品質だが `linear` より計算量が大きい。
- 長尺・高密度入力では `--range` 分割、`--preset aggressive`、`--max-segment-frames` 調整が運用回避策になる。
- `shakevmd --smooth` は aggressive 相当 + max5 を固定設定として使う。
- 既定設定で長時間未完了になる状態を性能回帰として扱う。

## 7. 検証計画

### 単体テスト

```powershell
pytest mmd_toolbox sparsevmd shakevmd
```

重点確認:

- 既知ベジェ曲線の復元。
- `_bezier_y_at()` と `_bezier_y_at_many()` の一致。
- カメラ位置・距離・FOV・回転の `bezier` 再構築。
- ボーン位置・回転の `bezier` 再構築。
- `linear` モードの既存挙動。
- strict / 非strict の分割挙動。
- `shakevmd --smooth` 既定 on と `--no-smooth` の差分。

### 性能ベンチ

最低限の比較:

| ケース | 目標 |
|---|---|
| `balanced/bezier --range 0:60` | Step A のベースラインから悪化しない |
| `balanced/bezier --range 0:180` | Step A のベースラインから悪化しない |
| `aggressive/bezier --max-segment-frames 5 --range 0:720` | Step A のベースラインから悪化しない |
| `aggressive/bezier --max-segment-frames 5` 全長 | 対話的に待てる範囲(目安 数分。§8) |
| `balanced/bezier` 全長 | 努力目標: 対話的に待てる範囲(目安 数分)。超えても追加最適化対象として次フェーズへ(必須でない) |
| `linear` 全長 | 高速な回避策として成立 |

### 品質比較

`--report-json` を使い、Step A で記録したベースライン(出力キー数・最大誤差・分割数等)と変更適用後を比較する。
report の位置誤差は軸別最大絶対誤差(`measure_*_errors`)なので、これは改修前後の相対悪化の検出に用いる。
プリセットの位置許容はユークリッド距離(`verify_*_track`)で意味が異なるため、許容適合の証明には使わない
(許容適合は `reduce_*_track` が呼ぶ出力後検証が担保する)。

- 出力キー数。
- 最大位置誤差。
- 最大距離誤差。
- 最大FOV誤差。
- 最大回転誤差。
- 分割数。
- 出力後検証で追加されたフレーム数。

## 8. 完了条件

以下を満たしたら本計画は完了とする。

- `pytest mmd_toolbox sparsevmd shakevmd` が成功する。
- `sparsevmd ... --preset aggressive --max-segment-frames 5`(全長)と `shakevmd` 既定 `--smooth`(代表素材・実出力。
  `--dry-run` では smooth 経路を測れない)が、Step A で素材ごとに設定した受入上限(対話的に待てる秒数。目安 数分)
  以内で完了し、かつ Step A 比(同手順の変更前後)で相対悪化しない。ベンチ入力は追跡外・環境依存のため計画書に
  一律の絶対秒数は固定せず(§3)、受入上限は着手者が Step A で自分の環境・素材に対して決めて記録する。
  代表素材は手元の実利用カメラ VMD(追跡外)。
- `balanced/bezier` 全長は努力目標(必須完了項目は上記 aggressive/max5 全長と代表素材)。実測を記録し、
  受入上限(§7)を超える場合は追加最適化を次フェーズ課題として具体策を残す。
- ベクトル化で品質が悪化しない: 出力キー数が変更前と一致し(キー数一致は必要条件。微小数値差で採否や制御点が
  変わりうるので、`--report-json` の軸別最大誤差も Step A のベースラインから悪化しないことを併せて確認する)。
  許容適合は `reduce_*_track` の出力後検証(位置はユークリッド距離、回転はカメラ=軸別角度誤差の最大・
  ボーン=quaternion 角度距離でプリセット許容内)が担保する。
- 出力後検証ループの反復状況を diagnostics/report から確認できる。
- 仕様書または本計画書に、性能上の制約と推奨運用が記録されている。

## 9. リスクと対策

| リスク | 対策 |
|---|---|
| ベクトル化で `_solve_factor` と微小差が出る | 単点版との比較テストを追加し、採否は量子化後再評価で確認する |
| `max_nfev` により制御点品質が下がる | 最後の手段にし、既知ベジェ復元と実データ誤差を比較する |
| 初期値削減でキー数が増える | 早期終了条件を許容内判定にし、許容超過時だけ追加初期値を試す |
| verification loop が依然として重い | 反復回数と追加フレーム数を diagnostics に出し、悪化箇所を特定できるようにする |
| aggressive/max5 が素材によって遅い | 実利用素材のベンチを追加し、`shakevmd` 側の固定値を再検討する |

## 10. 作業順序

1. 現状ベースラインを起点に、全長ベンチを記録する。
2. Step C の diagnostics を追加し、出力後検証の実態を見える化する。
3. Step B の `_bezier_y_at_many()` を実装する。
4. テストと短・中・全長ベンチで速度と品質を比較する。
5. 改善が不足する場合のみ Step D の `max_nfev` / 初期値制御を検討する。
6. Step E の仕様追記を行う。

この順序なら、すでに完了したキャッシュ改善を前提に、残っている支配項だけを小さく潰せる。
