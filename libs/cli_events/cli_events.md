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
- argparse の使用法エラーを各 CLI が引き取れるようにする。argparse が自ら標準エラーへ出して終了する
  代わりに例外を送出する ArgumentParser と、その例外を `error` イベントへ変換するヘルパを提供し、
  構造化出力モードではエラーイベントへ、それ以外では規約 [§6.1](../../docs/conventions/cli-interface.md#61-人間向け表示の表記) のエラー行へ振り替えられるようにする。
- 失敗報告の共通経路を提供する。構造化出力モードの `error` イベントと、人間向けの
  エラー行(規約 [§6.1](../../docs/conventions/cli-interface.md#61-人間向け表示の表記))の両方を 1 つの入口から出し、標準出力へ書けない場合の後退
  (規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力) が定める終端保証の例外)まで含めて各 CLI が同じ挙動になるようにする([§6](#6-失敗報告ヘルパ))。
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

- ストリームは `result` または `error` の**ちょうど 1 つ**で終端する(規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力)。同節が定める終端保証の
  例外を除く)。
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
  JSON Lines を壊さないため)。1 ストリームにつき終端イベントはちょうど 1 つ(規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力) が定める
  終端保証の例外を除く。標準出力への書き込み自体が失敗した場合の後退は [§6](#6-失敗報告ヘルパ))。

### 3.1 公開 API

- `EventEmitter(stream)`: `progress` / `warning` / `result` / `error` の各メソッド(`**fields` でペイロードを
  受ける)と `terminated` プロパティを持つ。`result` / `error` の後の送出は `StreamTerminatedError` を送出する。
- `error_event(*, code, message, exit_code, field=None, path=None)`: `error` イベントの dict ビルダー
  (`type:"error"` 固定、`field`/`path` は既定 `None`)。返り値はそのまま `EventEmitter.error(**event)` へ展開できる。
- `EVENT_TYPES`: イベント種別の語彙タプル `("progress", "warning", "result", "error")`。

## 4. argparse エラー変換ヘルパ

argparse は既定で使用法エラーを用法(usage)込みで標準エラーへ出して終了する。各 CLI はこの経路を自分で
引き取る必要がある: 構造化出力モードでは `error` イベントへ振り替え、それ以外でも規約 [§6.1](../../docs/conventions/cli-interface.md#61-人間向け表示の表記) のエラー行
1 行へ揃える。本モジュールは、argparse が終了する代わりに例外を送出する ArgumentParser と、その例外
(メッセージと該当引数)を `error` イベントへ変換するヘルパ、argparse の文言から該当引数名(`field`)を
取り出すヘルパを提供する。argparse のエラーは引数エラーなので、**終了コードは規約
[§5](../../docs/conventions/cli-interface.md#5-エラーと終了コード) の基底共通の `2` で固定**する(基底 `0`〜`3` の意味は全ツール共通で各ツール裁量にしない。
規約 [§6](../../docs/conventions/cli-interface.md#6-横断的な一貫性))。安定 `code` 値と該当 `field` は各ツールが定めるのでヘルパは呼び出し側から
受け取って載せ(具体の `code` 値は各ツールの仕様書が定める)、`message` は例外が保持する argparse 生成の
文言をそのまま載せる(翻訳・再構築しない。規約 [§6.1](../../docs/conventions/cli-interface.md#61-人間向け表示の表記) が argparse 自身の固定文言を英語のままと定める)。

公開 API:

- `MachineArgumentParser`(`argparse.ArgumentParser` のサブクラス): 使用法エラー時に標準エラーへ出して終了
  する代わりに `ArgumentParseError`(`message` 保持)を送出する。**構造化出力モードに限らず全経路で使う**
  (どちらのモードでも argparse 既定の出力・終了を各 CLI が引き取るため)。`--help` / `--version` は
  `error()` を経由しないため影響を受けない(規約 [§3](../../docs/conventions/cli-interface.md#3-機械モードの起動) のメタ操作の例外)。
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

## 6. 失敗報告ヘルパ

失敗の報告は「構造化出力モードなら `error` イベントで終端し、そうでなければ人間向けのエラー行を
標準エラーへ 1 行出す」という同じ形を全 CLI が持つ。この分岐と、標準出力へ書けない場合の後退
(規約 [§4](../../docs/conventions/cli-interface.md#4-イベントストリーム標準出力) が定める終端保証の例外)を本モジュールへ集約し、各 CLI は結果の終了コードを
そのまま返すだけにする。

公開 API:

- `emit_failure(emitter, *, code, message, exit_code, field=None, path=None, stderr=None, **extra) -> int`:
  失敗を 1 回だけ報告し、受け取った `exit_code` をそのまま返す(呼び出し側は `main()` の戻り値へ回せる)。
  挙動:
  - `emitter` が `None`(構造化出力モードでない・エミッタ未確立)か既に終端済みなら、人間向けの
    エラー行(規約 [§6.1](../../docs/conventions/cli-interface.md#61-人間向け表示の表記) の `error: <本文>`)を 1 行出す。終端済みの場合、既に出した終端
    イベントは書き換えず、返り値は今回受け取った `exit_code` とする(result 送出の直後に中断・想定外
    例外が起きた場合の二次的な失敗を握り潰さないため。この場合だけはイベントストリームの終端種別と
    プロセスの終了コードが一致しない)。
  - それ以外は `error` イベント([§3.1](#31-公開-api) の `error_event` の形)を送出して終端する。この送出が
    `OSError` または `ValueError` になった場合は同じ人間向けのエラー行 1 行へ切り替え、例外を呼び出し側へ
    漏らさない。この 2 種を対象にするのは、書き込み先が使えないときの例外が両方の形で現れるため
    (呼び出し側がパイプを先に閉じた場合は `OSError`、標準出力自体が閉じられている場合は `ValueError`)。
  - エラー行の書き込みが失敗した場合はその失敗を握り潰し、終了コード・出力を変えない
    (人間向け標準エラーの書き込み失敗でプロセスを落とさない。規約 [§10](../../docs/conventions/cli-interface.md#10-移植性パス符号化改行ロケール))。
  - `**extra` は `error` イベントへ載せる追加キー(ツール固有のペイロード。例: 失敗ステージの id)。
    人間向けのエラー行には載せない。基底のイベントへ後から重ねるので、`error_event` が定める `type` と
    同名のキーを渡してはならない(渡した場合は呼び出し側の誤り。`code` / `message` / `exit_code` /
    `field` / `path` は名前付き引数なので `**extra` 経由では渡せない)。
  - `stderr` はエラー行の出力先。既定の `None` は**呼び出しの時点で**標準エラーへ解決する
    (定義時に束縛すると、テストの差し替え・捕捉が効かなくなる)。

## 7. テスト

`pytest libs/cli_events` で単体検証する(テスト方針は [../vmd/vmd.md §4](../vmd/vmd.md#4-テスト方針) に準ずる。決定論的・外部依存なし)。

- エミッタ出力が有効な JSON Lines(各行が単一 JSON)で、UTF-8 で書かれること。
- エミッタ出力の行区切りが LF(`\n`)で、`\r\n` を含まないこと(バイト列で検証。規約 [§10](../../docs/conventions/cli-interface.md#10-移植性パス符号化改行ロケール))。
- `result` / `error` 送出後に追加送出を許さないこと(終端規則)。
- argparse エラー変換ヘルパが `type:"error"` のイベントを生成すること。
- argparse 文言からの `field` 抽出([§4](#4-argparse-エラー変換ヘルパ))が規則どおり(`argument` / `unrecognized` / `required` の 3 形式と、
  いずれにも当たらない文言の `None` フォールバック)であること。
- SIGBREAK 橋渡しヘルパ([§5](#5-中断シグナルの橋渡し))が、Windows ではハンドラ登録後に `KeyboardInterrupt` を送出し、Windows
  以外では no-op であること。
- 失敗報告ヘルパ([§6](#6-失敗報告ヘルパ))が、エミッタ無し・終端済み・`error` 送出が `OSError` / `ValueError` に
  なる場合のそれぞれで出力先(イベント / 人間向けエラー行)を切り替え、受け取った終了コードを返すこと。
  エラー行の書き込みも失敗する場合に例外を漏らさないこと。
- 実子プロセスへの実際の `CTRL_BREAK_EVENT` 配送は、OS 配送自体が非決定的なため([§5](#5-中断シグナルの橋渡し))通常スイートには
  含めず、手動診断用として別途用意する。
