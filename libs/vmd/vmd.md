# vmd — VMD読み書き・補間評価・カメラ変換・キーフレーム疎化

層: **フォーマット層**([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))。

VMD(MMD のモーション・カメラの形式)を読み書きし、MMD 本体と同じ結果になる補間曲線の評価、カメラの
座標変換、キーフレームの疎化を提供する。バイナリレイアウトの正本は
[VMD_file_format.md](../../docs/specs/vmd/VMD_file_format.md)。

## 下位仕様書

読み書きは [vmd-io.md](vmd-io.md)、補間曲線の評価は [vmd-interp.md](vmd-interp.md)、カメラの座標変換は
[vmd-camera.md](vmd-camera.md)、キーフレームの疎化は [vmd-reduce.md](vmd-reduce.md) を、それぞれの正本とする。

## 時間軸

VMD のキーはフレーム番号だけを持ち、fps を持たない。本リポジトリではフレーム 30 を 1 秒とみなす。

## 対象範囲

- 揺れの生成やカット検出の閾値の決め方など、個々のツールの判断は持たない。
- レンダリング・物理演算・IK は評価しない。

## 受入条件

補間評価とカメラ変換が MMD 本体と一致することは、元の動きを変えずにこれらを通して書き出した VMD と
元の VMD を MMD で再生し、見た目に差が無いことを人が一度確かめて受け入れる。
