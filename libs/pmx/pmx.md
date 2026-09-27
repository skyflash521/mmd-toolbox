# pmx — PMX読み取り・FK評価

層: **フォーマット層**([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))。

PMX モデルからボーン階層を読み取り、VMD のボーンキーをその階層へ適用する前方運動学(FK)評価を
提供する。バイナリレイアウトの正本は [PMX仕様](../../docs/specs/pmx/PMX仕様.txt)。

## 対象範囲

- 読み取り専用で、PMX の編集・書き出しは行わない。
- FK は近似で、IK・付与親・物理演算は評価しない。
