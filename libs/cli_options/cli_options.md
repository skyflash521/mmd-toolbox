# cli_options 仕様書

層: **共有ドメイン層**([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))。

CLI ツールの数値引数と複合トークン引数について、受理範囲・書式を1か所で定義し、その定義から argparse の
型検証と、自己記述で公開する制約の双方を導く。自己記述の options 配列は parser の登録内容から組み立てる。

## 対象範囲

担うのは値の形(型・範囲・書式)まで。
