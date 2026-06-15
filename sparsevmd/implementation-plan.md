# sparsevmd 実装計画

この文書は `sparsevmd/sparsevmd.md` をもとにした実装計画である。

## 前提

- 実装対象は `sparsevmd/` 配下と、必要最小限のパッケージ設定に限定する。
- `shakevmd/` 配下の作業中ファイルには触れない。
- 現在 `sparsevmd/` は `.git/info/exclude` でローカル除外中。
  正式にGit管理へ移す段階で `sparsevmd/` の除外を解除する。
- 補間評価は `mmd_toolbox.vmd.interp` に委譲し、sparsevmd側で再実装しない。
  camera全チャンネルの列サンプリングは、`sample_camera` が単一フレームAPIであるため、
  sparsevmd側でフレームループを回して作る。
- VMD読み書き・正規化は `mmd_toolbox.vmd.io` / `mmd_toolbox.vmd.types` に委譲する。

## 既存構成への影響

- `pyproject.toml` は現状 packages.find が `mmd_toolbox*` / `shakevmd*` のみ、
  `[project.scripts]` が `shakevmd` のみ。sparsevmd実装時は packages.find に
  `sparsevmd*` を、`[project.scripts]` に `sparsevmd = "sparsevmd.cli:main"` を追加する。
- pytest対象も現状 `mmd_toolbox` / `shakevmd` のみ。
  sparsevmd実装時は `sparsevmd` を追加する。
- `sparsevmd/` が `.git/info/exclude` で除外されたまま `pyproject.toml` だけを
  更新すると、配布物に含まれない齟齬が起きる。
  除外解除、パッケージ対象追加、pytest対象追加は同じ作業単位で行う。
- 仕様上は `scipy.optimize.least_squares` を使うため、
  ベジェフィット実装時に `scipy` を依存へ追加する。

## Phase 0: 仕様の最終確認

- カメラ `perspective` フラグの出力方針は `sparsevmd.md` §4.2 / §6.1 で確定済み。
  「該当フレーム以前の直近キー値をホールドし、切り替えフレームは
  `--no-cut-detect` でも無効化しない常時境界とする」。実装で踏襲する。
- ボーン補間64バイト、カメラ補間24バイトの生成API方針を決める。
  現状 mmd_toolbox には制御点から補間バイト列を組み立てるwriter APIがない。
  Phase 6/8 のブロッカーとして、生成処理を mmd_toolbox 側へ置くか
  sparsevmd/fit.py 側へ置くかをPhase 0で確定する。
- ボーン補間64バイト生成では、Byte[2]/Byte[3] とシフトコピー側の
  Byte[17]/Byte[18] の両方を仕様通り書く必要がある。
  物理フラグ上書き仕様と衝突しないテスト方針もここで決める。
- 対象外セクションのバイト単位保持が `io.read -> io.write` の
  ラウンドトリップ保証で本当に達成できるか、早期に既存テスト/追加テストで確認する。
- `sparsevmd/` をGit追跡するタイミングで `.git/info/exclude` の `sparsevmd/` を解除する。

Phase 0完了条件:

- `perspective` フラグ出力方針が仕様書に反映されている(§4.2 / §6.1、反映済み)。
- 補間バイト生成処理の配置が決まっている。
- bone補間64バイト生成のテスト方針が決まっている。
- 対象外セクションのラウンドトリップ前提が確認済み。
- `sparsevmd/` のGit除外解除タイミングが確定している。

## Phase 1: パッケージ土台

作成候補:

- `sparsevmd/__init__.py`
- `sparsevmd/cli.py`
- `sparsevmd/presets.py`
- `sparsevmd/report.py`

実施内容:

- CLIの引数定義と検証を作る。
- `--preset precise|balanced|aggressive` からtol群へ展開する。
- 個別tolが明示された場合はプリセット値より優先する。
- `--min-segment-frames`
- `--max-segment-frames`
- `--strict`
- `--verbose`
- `--list-bones` を実装する。
- `--dry-run` の骨格を作る。
- `.git/info/exclude` の `sparsevmd/` 解除と同じ作業単位で、
  `pyproject.toml` に `sparsevmd*`(packages.find)とpytest対象(testpaths)を追加し、
  `[project.scripts]` に console script `sparsevmd = "sparsevmd.cli:main"` を追加する
  (sparsevmd.md §2.1 の `sparsevmd INPUT.vmd` コマンドをインストールするため)。

## Phase 2: 入力・選択・範囲解決

作成候補:

- `sparsevmd/selection.py`
- `sparsevmd/ranges.py`

実施内容:

- `--bone`
- `--bone-glob`
- `--bone-group`
- `--bone-file`
- `--exclude-bone`
- `--exclude-bone-glob`
- `--exclude-bone-group`
- `--list-bones`
- `START:END` / `START:` / `:END`

重要要件:

- ボーンセクションにキーが存在するのに選択結果が0件の場合は終了コード2
  (sparsevmd.md §2.2)。ボーンセクションが空の場合は選択0件ではなくキー不在として
  扱い、下記の対象セクション不在ルール(target all のフォールバック含む)に従う。
- 対象セクションにキーが存在しない場合は終了コード1。
  ただし `--target all` では片方のセクションが空でも他方にキーがあれば、
  存在する側のみ処理して続行する(camera0件+boneあり→boneのみ、
  bone0件+cameraあり→cameraのみ。sparsevmd.md §3.1)。コード1は
  `--target` が対象とするセクションのキーが1件も無い場合に限る。
  また `--list-bones` は検査モードであり、VMD読み込みに成功すれば
  対象セクションのキー有無やボーン選択結果に関わらず終了コード0とする
  (sparsevmd.md §2.7)。
- `--range` の省略端は対象トラック全体の最小/最大フレームで1回だけ展開する。
- 各トラックの実処理範囲はグローバル範囲との積集合にする。
- 全トラックの積集合が空なら正常終了し、「削減対象なし」と記録する。

## Phase 3: サンプリング層

作成候補:

- `sparsevmd/sample.py`

実施内容:

- VMDを読み込む。
- cameraを1トラック化する。
- boneをボーン名ごとのトラックに分割する。
- camera/boneのみ内部作業ビューで正規化する。
- スカラーチャンネルは `sample_range` に委譲してサンプル列を作る。
  戻り値はnumpy配列である前提で、誤差評価層との型契約を定める。
- camera全チャンネルのサンプル列は、`sample_camera` を各フレームで呼ぶ
  ループをsparsevmd側で持つ。
  camera回転は `sample_camera(keys, frame)["rotation"]` のEuler 3要素tupleとして扱う。
- perspective は補間チャンネルではなく、`sample_camera` の戻り値にも含まれない
  (distance/position/rotation/fov のみ)。`CameraKey.perspective` を入力キー列から
  直接読み、当該フレーム以前で最も近いキーの値をホールドする(sparsevmd.md §4.2)。
  cut検出(Phase 4)は perspective が切り替わるフレームを常時境界として追加する。
- bone回転のサンプル列は `sample_range(keys, "rot", ...)` を使う。
  戻り値はクォータニオンtupleのリストである前提で扱う。

検証:

- まだ削減は行わず、入力サンプル列が正しく得られることをテストする。

## Phase 4: 不連続検出

作成候補:

- `sparsevmd/cuts.py`

実施内容:

- camera: `POS,ROT,DIST`
- bone: `POS,ROT`
- `--no-cut-detect`
- `--keep-frame`
- range端
- cut境界

重要要件:

- カメラ距離ジャンプは `DIST` で判定する。
- ボーン位置差はMMD距離単位、回転差は度で判定する。
- 不連続境界をまたいで補間曲線を作らない。
- `--no-cut-detect` 指定時は閾値による自動不連続検出だけを無効化する。
  range端と `--keep-frame` は引き続き必須キーとして扱う。
- `--keep-frame` が全削減範囲外なら警告して無視する。
- 特定トラックの有効範囲外なら、そのトラックでだけ無視する。

## Phase 5: 最小削減器(linear mode)

作成候補:

- `sparsevmd/reduce.py`
- `sparsevmd/fit.py`

実施内容:

- 必須キー間を区間化する。
- `--max-segment-frames` で事前分割する。
- 誤差を満たせる範囲では `--min-segment-frames` を下回る分割を行わない。
- `--curve-mode linear` を先に実装する。
- 誤差超過時は最大誤差フレームで再帰分割する。
- 分割点選択は最大誤差フレームを基本にしつつ、局所極値と速度符号反転を
  優先候補として扱う。
- 非strictで `--min-segment-frames` 長まで分割しても誤差を満たせない区間は、
  仕様 sparsevmd.md §1.2 / §2.5 のとおり下限を無視して1フレームまで分割し保持する。
- `--strict` 指定時は、最小区間まで分割しても許容誤差を満たせない時点で
  下限無視の密フォールバックを行わず、終了コード4へつなぐエラーを返す。

検証:

- 連続フレームの線形移動が先頭/末尾キーと線形補間曲線に削減されること。
- まずカメラ位置などのスカラー値で確認する。
- `--strict` 指定時、許容誤差を満たせないケースでコード4になること。

## Phase 6: ベジェフィット

このPhaseで `fit.py` のフィット関数(スカラー・camera回転・bone回転)を実装する。
ただしreduceパイプラインへ接続するのはスカラー曲線フィットのみとする。
camera回転・bone回転のフィット関数はここで実装・単体検証まで行うが、
reduceパイプラインへの接続(回転区間での呼び出し有効化)はPhase 7のlinear mode
実装後に行う。

対象:

- `sparsevmd/fit.py`

実施内容:

- スカラー曲線フィットを実装し、reduceパイプラインへ接続する。
- camera回転用の共通曲線フィット関数を `fit.py` に実装する(この時点ではパイプライン
  未接続。単体検証のみ)。3軸Eulerを同時に評価し、1本の補間曲線で全軸の許容誤差を
  満たすか判定する。
- bone回転用の共通曲線フィット関数を `fit.py` に実装する(同上、パイプライン未接続)。
  slerp係数に適用する1本の補間曲線で角度距離の許容誤差を満たすか判定する。
- `scipy.optimize.least_squares` を使う。
- 制御点を `0..127` 整数へ量子化する。
- 量子化後に必ず再評価する。
- camera補間24バイトを生成する。
- bone補間64バイトを生成する。
- bone補間64バイト生成では、Byte[2]/Byte[3] だけでなく
  シフトコピー側の Byte[17]/Byte[18] へも対応する制御点を書き込む。
  生成処理の配置はPhase 0で決めた方針に従う。

検証:

- 既知ベジェ曲線で生成した密サンプルから、少数キーと近い制御点が得られること。
- bone補間64バイトの生成結果から `BoneKey.control_points()` で同じ制御点を復元できること。

## Phase 7: 回転対応

このPhaseでは回転チャンネルの誤差評価・分割・linear modeを先に実装する。
ベジェ曲線探索への接続はPhase 6のfit機能が入った後に有効化する。

camera rotation:

- Euler角を軸ごとにunwrapする。
- 3軸共通の補間曲線を使う。
- 最大角度誤差で判定する。
- linear modeで先に回転誤差評価と分割を通し、その後Phase 6のベジェ曲線探索へ接続する。

bone rotation:

- quaternionを同一半球に揃える。
- slerpを使う。
- 角度距離で誤差判定する。
- slerp軌道から外れる場合は分割する。
- 回転方向反転を分割点の優先候補として扱う。
- linear modeで先に回転誤差評価と分割を通し、その後Phase 6のベジェ曲線探索へ接続する。

検証:

- camera回転共有曲線。
- bone回転のslerp誤差。
- 回転方向反転や大円弧外サンプルの分割。

## Phase 8: 出力再構築

実施内容:

- 削減後キー列から `CameraKey` / `BoneKey` を生成する。
- 対象外セクションを透過する。ボーン選択で非選択のトラック・削減不能トラックは
  元値と補間曲線のまま出力キー列へ含める(sparsevmd.md §3.2)。
- `--range` 外の非隣接キーは元値と補間曲線を保持する。
- 境界キー追加時は隣接補間曲線の書き換えをreportに記録する。
- 出力直前に全対象トラックを再サンプリングして許容誤差を検証する。検証は
  VMD保存と同じ float32 量子化を適用した出力キー値で行う(sparsevmd.md §7.1)。
  実装上は `io.write` 相当でfloat32へ丸めたキー値、または書き出した出力VMDを
  読み直したキー値を再サンプリングする。
- 非strictでは、検証で誤差超過が残る区間をさらに分割する。
- strictでは、許容誤差を満たせない区間が残った場合に終了コード4へつなぐ。
- `write_file` で出力する。

注意:

- 対象セクションは再構築されるため、バイト一致は保証しない。
- 対象外セクションはバイト単位保持を目標にする。
- 対象外セクションのバイト保持は `io.read -> io.write` の保証に依存するため、
  Phase 0/Phase 10 のラウンドトリップ確認で早めにリスクを潰す。

## Phase 9: レポート

作成候補:

- `sparsevmd/report.py`

実施内容:

- `--dry-run`
- `--report-json`
- `--preview-csv`

重要要件:

- `--dry-run` は出力VMDだけを作らない。
- `--report-json` / `--preview-csv` はdry-run時も指定されていれば書く。
- レポート書き込み失敗は終了コード3。
- JSON/CSVの親ディレクトリ不存在もエラー。

JSONに含める候補:

- 選択ボーン
- トラック別入力キー数
- トラック別出力予定キー数
- 削減率
- 最大誤差
- 分割理由
- cut位置
- keep-frame
- range積集合空
- FOV量子化警告

## Phase 10: テスト仕上げ

テスト項目:

- CLI引数検証
- ボーン選択
- `--list-bones`
- range展開
- 全トラックrange積集合空
- `--keep-frame` 範囲外
- FOV量子化
- カメラ距離カット
- dry-run時のreport/csv出力
- 対象セクション0件
- ボーン選択結果0件
- 全トラック1キー以下
- strict終了コード4
- 再現性
- `io.read -> io.write` のラウンドトリップ確認により、対象外セクションの
  バイト単位保持が実現可能であること。
- bone補間64バイト生成結果が `BoneKey.control_points()` で復元可能であること。

## 推奨実装順

「Phase N」は機能領域ごとのまとまり(テーマ分類)であり、着手順ではない。
実際の着手順は本節の番号付きリストを正とする。Phase間の依存(例: Phase 6で
回転フィット関数を実装し、Phase 7のlinear mode後にreduceパイプラインへ接続する)は
この順序に展開済みであり、linear modeを各チャンネルへ通してから最後にベジェ
フィットを全チャンネルへ接続する。

1. `.git/info/exclude` 解除 + packaging/testpaths更新
2. CLI + presets + selection + range + list-bones
3. sample.py + dry-run統計
4. linear mode reducer + strict
5. camera scalar + FOV
6. camera rotation(linear mode)
7. bone position
8. bone rotation(linear mode)
9. bezier fitting + 補間バイト生成
10. report/csv

## 最初のMVP

最初のMVPは以下に限定する。

- `--target camera`
- `--curve-mode linear`
- `--dry-run`
- `--list-bones`
- `--range`
- `--report-json`
- `--preset balanced`
- ボーン選択オプション(`--bone` / `--bone-glob` / `--bone-group` / `--bone-file` /
  `--exclude-bone` / `--exclude-bone-glob` / `--exclude-bone-group`)。
  MVPではこれらの解析・検証と `--list-bones` への反映までを対象とし、
  ボーンの実削減は行わない。`--target camera` との同時指定は仕様どおりエラー
  (終了コード2)とするため、選択状態の確認は `--list-bones` 経由で行う。

目的:

- 入力
- ボーン/カメラ対象判定
- range解決
- サンプリング
- 誤差評価
- dry-run/report

MVPでは `--target camera` のため、ボーンは入力有無・選択ルール・一覧表示などの
判定系だけを確認し、ボーンの実削減は行わない。

この骨格を先に固め、その後にベジェフィットとボーン回転を載せる。
