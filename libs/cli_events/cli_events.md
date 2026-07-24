# cli_events 仕様書

CLI ツールの機械モード出力(JSON Lines イベントストリーム)を送出する共有基盤ライブラリ。

層: **共有ドメイン層**([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))。複数の CLI ツールが共有する CLI 出力契約
(機械モードのイベント送出)を、特定形式の入出力でも単一ツール固有でもないため共有ドメイン層に置く。

依存: Python 標準ライブラリのみ(`json` / `argparse` 連携)。フォーマット層・他の共有ドメイン層へは依存しない
(テスト専用の `cli_events.testing` は例外で、`pytest` に依存する。[§5](#5-中断シグナルの橋渡し))。

正本関係: 機械モードの契約(イベント種別の語彙・終端規則・チャネル固定・終了コードの基底・stdout の
UTF-8 固定)は[CLI インターフェース規約](../../docs/conventions/cli-interface.md)が正
(**以下、本書で「規約」はこの CLI インターフェース規約 [`docs/conventions/cli-interface.md`](../../docs/conventions/cli-interface.md) を指す**。
本書中の「規約 §N」はこの文書の N 節)。本書はその契約を実装する共有モジュールの責務・API・挙動を
定める(規約が正、本書は実装ビュー)。各ツール固有のイベントペイロード(統計・`code` 値など)は
各ツールの仕様書が定め、本書・本モジュールはペイロードの中身を固定しない。

---

## 1. 責務

- 機械モードの標準出力を **JSON Lines**(1 行 1 オブジェクト、UTF-8)として送出する共通エミッタを提供する。
- イベント種別の語彙(`progress` / `warning` / `result` / `error`)を一元的に定義し、各 CLI が個別に JSON を
  組み立てないようにする(各ツールでの重複実装と語彙のずれを防ぐ。規約 [§6](../../docs/conventions/cli-interface.md#6-横断的な一貫性) の横断一貫性に対応)。
- argparse のエラーを `error` イベントへ変換するヘルパを提供し、各 CLI が argparse のエラー出力経路を
  機械モードのエラーイベントへ振り替えられるようにする。
- OS依存の協調的中断シグナルを `KeyboardInterrupt` へ橋渡しするヘルパを提供し、各 CLI が既存の
  `except KeyboardInterrupt` 経路(規約 [§8](../../docs/conventions/cli-interface.md#8-キャンセルと出力の原子性) の `cancelled` エラーイベント終端)で、プラットフォームを
  問わず中断を捕捉できるようにする(OS のイベント配送自体に起因する残存限界は [§5](#5-中断シグナルの橋渡し))。

本モジュールはイベントの**封筒(`type` と送出・終端規則)**を担い、各イベントの**中身(ペイロードのキー)は
呼び出し側(各ツール)が決める**。規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力) の「種別の語彙と終端規則だけを共通化し、ペイロードは各ツールが
定める」に対応する。

## 2. イベント種別

規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力) の 4 種を `type` フィールドで表す。各種別の役割は規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力) が正:

| `type` | 役割 | 終端 |
|---|---|---|
| `progress` | 重い処理の進行 | 非終端 |
| `warning` | 継続可能な問題 | 非終端 |
| `result` | 正常終了の要約 | **終端** |
| `error` | 失敗の理由 | **終端** |

- ストリームは `result` または `error` の**ちょうど 1 つ**で終端する(規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力))。
- 各イベントは少なくとも `type` を持つ。`type` 以外のキー(ペイロード)は呼び出し側が渡し、本モジュールは
  その内容を解釈・検証しない(ペイロードは各ツール仕様書が定める)。

## 3. エミッタ

エミッタは出力ストリーム(機械モードでは標準出力)を受け取り、各種別のイベントを 1 行ずつ送出する。

- **符号化**: エミッタは UTF-8 バイトを書き込むバイナリストリーム(機械モードでは標準出力のバイナリ
  バッファ)を受け取り、各行を UTF-8 で書く。プラットフォームのロケール符号化(Windows の cp932 等)に
  依存させず、リダイレクト・パイプでもロケール外の文字で符号化に失敗して途切れさせない(規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力))。
  非 ASCII 文字は `\uXXXX` へエスケープせず UTF-8 のまま出す(`json.dumps(ensure_ascii=False)` 相当)。
- **1 行 1 オブジェクト**: 各イベントを単一行の JSON として書き、行末で改行する。行区切りは **LF(`\n`)固定**
  とし、バイナリストリームへ書くことでプラットフォームの改行変換(Windows のテキストモードの CRLF 変換)を
  避け `\r` を混入させない(規約 [§10](../../docs/conventions/cli-interface.md#10-移植性パス符号化改行ロケール))。JSON 内に生の改行を入れず、呼び出し側が逐次パースできるようにする。
- **終端規則の保証**: `result` / `error` を送出したら、それ以降のイベント送出を許さない(終端後の追記で
  JSON Lines を壊さないため)。1 ストリームにつき終端イベントはちょうど 1 つ。

### 3.1 公開 API

- `EventEmitter(stream)`: `progress` / `warning` / `result` / `error` の各メソッド(`**fields` でペイロードを
  受ける)と `terminated` プロパティを持つ。`result` / `error` の後の送出は `StreamTerminatedError` を送出する。
- `error_event(*, code, message, exit_code, field=None, path=None)`: `error` イベントの dict ビルダー
  (`type:"error"` 固定、`field`/`path` は既定 `None`)。返り値はそのまま `EventEmitter.error(**event)` へ展開できる。
- `EVENT_TYPES`: イベント種別の語彙タプル `("progress", "warning", "result", "error")`。

## 4. argparse エラー変換ヘルパ

argparse は既定で使用法エラーを標準エラーへ出して終了する。機械モードではこの経路を `error` イベントへ
振り替える必要がある。本モジュールは、argparse のエラー(メッセージと該当引数)を `error` イベントへ
変換するヘルパと、argparse の文言から該当引数名(`field`)を取り出すヘルパを提供する。argparse のエラーは
引数エラーなので、**終了コードは規約 [§5](../../docs/conventions/cli-interface.md#5-エラーと終了コード) の基底共通の `2` で固定**する(基底 `0`〜`3` の意味は全ツール共通で
各ツール裁量にしない。規約 [§6](../../docs/conventions/cli-interface.md#6-横断的な一貫性))。一方、安定 `code` 値・該当 `field`・`message` は各ツールが定めるので、
ヘルパはそれらを呼び出し側から受け取って載せる(具体の `code` 値は各ツールの仕様書が定める)。

公開 API:

- `MachineArgumentParser`(`argparse.ArgumentParser` のサブクラス): 使用法エラー時に標準エラーへ出して終了
  する代わりに `ArgumentParseError`(`message` 保持)を送出する。`--help` / `--version` は `error()` を経由
  しないため影響を受けない(規約 [§3](../../docs/conventions/cli-interface.md#3-機械モードの起動) のメタ操作の例外)。
- `argparse_error_event(error, *, code, field=None)`: `ArgumentParseError` を `error` イベントへ変換し、
  `exit_code` は基底共通の `2` で固定する(`code`/`field` は呼び出し側が渡す)。
- `argparse_error_field(message)`: argparse の使用法エラー文言から対象引数名(`error` イベントの `field`)を
  ベストエフォート抽出する。argparse は起因引数を構造化して渡さないため、標準の文言形から次の規則で取り出す
  (いずれにも当たらなければ `None`。詳細は `message` 側に残す):
  - 「`argument <引数名>: `」で始まる文言 → コロン前の引数名。複数のオプション文字列が「`/`」で連結される
    (例 `-o/--output`)場合は最後(長形式)を採り、先頭が「`-`」でない positional 名はそのまま返す。
  - 「`unrecognized arguments: `」で始まる文言 → 続くトークン列の最初の 1 語。
  - 「`the following arguments are required: `」で始まる文言 → 続く名前列の先頭(カンマ区切りの最初)。
  - いずれにも当たらない文言 → `None`。

## 5. 中断シグナルの橋渡し

規約 [§8](../../docs/conventions/cli-interface.md#8-キャンセルと出力の原子性) の協調的な中断は、各 CLI が `KeyboardInterrupt` を捕捉して `cancelled` エラーイベントへ変換する
形で実装する([§1](#1-責務))。CPython は SIGINT(Ctrl-C)には既定で `KeyboardInterrupt` を送出するハンドラを
持つが、Windows の `CTRL_BREAK_EVENT`(SIGBREAK)には持たない。呼び出し側(GUI 等)が
`CREATE_NEW_PROCESS_GROUP` で起動した子プロセスへは `CTRL_C_EVENT` を送れず `CTRL_BREAK_EVENT` のみが
使えるため、このハンドラを登録しない CLI は Windows からの中断要求で、Python の例外処理を経ないまま
OS の既定動作(`STATUS_CONTROL_C_EXIT`)により即座に終了し、`cancelled` エラーイベントも出せない。

公開 API:

- `install_sigbreak_handler()`: Windows かつ SIGBREAK が存在する環境でのみ、SIGBREAK を
  `KeyboardInterrupt` へ変換するハンドラを登録する。それ以外の環境では何もしないため、各 CLI は
  プラットフォームを判定せず無条件に呼べる。登録が有効になった時点で構造化出力(機械モードの
  `EventEmitter` 等)が既に組み上がっている必要があるため、呼ぶ順序は「構造化出力の準備が済んだ後・
  本体処理より前」とする。

`install_sigbreak_handler()` は呼び出されたプロセスの SIGBREAK ハンドラを書き換える。同一プロセス内で
これを繰り返し呼ぶテスト(CLI の `main()` を直接呼ぶテスト等)がテスト間でハンドラを引き継がないよう、
`cli_events.testing` が pytest フィクスチャ `restore_sigbreak_handler`(各テストの前後で SIGBREAK
ハンドラを保存・復元する。Windows 以外・SIGBREAK 非搭載環境では何もしない)を提供する。
`install_sigbreak_handler()` を呼ぶ各 CLI ツールのテストは、このフィクスチャを利用してテスト間の
副作用漏れを防ぐ。

**既知の残存限界**: ハンドラを登録していても、環境によっては `CTRL_BREAK_EVENT` の配送の一部で
`cancelled` エラーイベントを出せないままプロセスが終了する場合がある(頻度は環境依存で、無条件に
発生するわけではない)。この場合、呼び出し側からは規約 [§8](../../docs/conventions/cli-interface.md#8-キャンセルと出力の原子性) の「呼び出し側がプロセスを強制終了した
場合」と区別が付かないため、新たな契約は設けず同じ扱い(終端イベント無しの途切れを中断として扱う)
で足りる。

## 6. テスト

`pytest libs/cli_events` で単体検証する(テスト方針は [../vmd/vmd.md §4](../vmd/vmd.md#4-テスト方針) に準ずる。決定論的・外部依存なし)。

- エミッタ出力が有効な JSON Lines(各行が単一 JSON)で、UTF-8 で書かれること。
- エミッタ出力の行区切りが LF(`\n`)で、`\r\n` を含まないこと(バイト列で検証。規約 [§10](../../docs/conventions/cli-interface.md#10-移植性パス符号化改行ロケール))。
- `result` / `error` 送出後に追加送出を許さないこと(終端規則)。
- argparse エラー変換ヘルパが `type:"error"` のイベントを生成すること。
- argparse 文言からの `field` 抽出([§4](#4-argparse-エラー変換ヘルパ))が規則どおり(`argument` / `unrecognized` / `required` の 3 形式と、
  いずれにも当たらない文言の `None` フォールバック)であること。
- SIGBREAK 橋渡しヘルパ([§5](#5-中断シグナルの橋渡し))が、Windows ではハンドラ登録後に `KeyboardInterrupt` を送出し、Windows
  以外では no-op であること。
- 実子プロセスへの実際の `CTRL_BREAK_EVENT` 配送は、OS 配送自体が非決定的なため([§5](#5-中断シグナルの橋渡し))通常スイートには
  含めず、手動診断用として別途用意する。
