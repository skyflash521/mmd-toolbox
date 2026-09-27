# cli_progress_router 仕様書

層: **共有ドメイン層**([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))。

進捗の報告を、機械モードでは progress イベントの送出へ、それ以外では人間向けライブ表示へ振り分ける。
振り分けが満たす進捗表示の外部契約は
[CLI インターフェース規約 §6](../../docs/conventions/cli-interface.md#6-横断的な一貫性) が正本。
