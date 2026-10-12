_DEFAULT_HELPS = {
    "--version": "バージョンを表示して終了する",
    "--machine": "出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                 "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない",
    "--describe": "オプション定義とプリセット一覧を JSON Lines の result で出力して終了する"
                  "(入力を読まない・入力不要の自己記述)",
    "--overwrite": "出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)",
    "--dry-run": "出力せず処理計画と診断を表示する(引数検証は実施する)",
    "--verbose": "通常実行でも診断を表示する(出力は書く)",
    "--quiet": "進捗表示を抑制する(警告・診断・終了コードは抑制しない)",
}


def add_common_arguments(parser, *, version, output_help, quiet=True, help_overrides=None):
    overrides = dict(help_overrides or {})
    unknown = sorted(set(overrides) - set(_DEFAULT_HELPS) - {"--output"})
    if unknown:
        raise ValueError(f"help_overrides に未知のフラグがある: {', '.join(unknown)}")

    def help_of(flag):
        return overrides.get(flag, _DEFAULT_HELPS[flag])

    parser.add_argument("--version", action="version", version=f"{parser.prog} {version}",
                        help=help_of("--version"))
    parser.add_argument("--machine", action="store_true", help=help_of("--machine"))
    parser.add_argument("--describe", action="store_true", help=help_of("--describe"))
    parser.add_argument("-o", "--output", help=overrides.get("--output", output_help))
    parser.add_argument("--overwrite", action="store_true", help=help_of("--overwrite"))
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help=help_of("--dry-run"))
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help=help_of("--verbose"))
    if quiet:
        parser.add_argument("--quiet", dest="quiet", action="store_true", help=help_of("--quiet"))
