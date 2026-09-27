# cli_events 仕様書

層: **共有ドメイン層**([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))。

CLI ツールが共有する、機械モードのイベント送出、argparse の使用法エラーの引き取り、失敗報告の経路、
Windows の `CTRL_BREAK_EVENT` を `KeyboardInterrupt` へ橋渡しする処理を提供する。機械モードの出力契約の
正本は [CLI インターフェース規約](../../docs/conventions/cli-interface.md)。

## 既知の限界

`CTRL_BREAK_EVENT` を橋渡ししていても、OS からの配送によっては `cancelled` エラーイベントを出せないまま
プロセスが終了することがある。呼び出し側からは
[CLI インターフェース規約 §8](../../docs/conventions/cli-interface.md#8-キャンセルと出力の原子性) の強制終了と区別できず、
同じ扱いになる。
