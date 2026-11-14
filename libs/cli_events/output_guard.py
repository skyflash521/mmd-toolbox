import os


def guard_output(output, *, overwrite, fail):
    if os.path.isdir(output):
        return fail("output_is_directory",
                    f"出力先がディレクトリです(ファイルパスを指定): {output}",
                    2, field="--output", path=output)
    if not overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります。上書きには --overwrite が必要: {output}",
                    2, field="--output")
    return None


def default_output_path(input_path, suffix):
    base, _ = os.path.splitext(input_path)
    return base + suffix
