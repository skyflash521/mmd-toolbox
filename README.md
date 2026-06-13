# mmd-toolbox

MikuMikuDance (MMD) 向けのユーティリティツール群。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `docs/specs/` | MMDファイル形式の詳細仕様(VMD/PMX)。バイナリレイアウトの正 |
| `mmd_toolbox/` | 共通ライブラリ。VMD読み書き・補間曲線評価・カメラ座標変換 |
| `shakevmd/` | カメラVMDに手ぶれを焼き込むCLIツール |

仕様書は各ディレクトリ直下に置かれている:

- [mmd_toolbox/mmd_toolbox.md](mmd_toolbox/mmd_toolbox.md) — 共通ライブラリ全体
  - [mmd_toolbox/vmd-io.md](mmd_toolbox/vmd-io.md) — VMD読み書き・正規化
  - [mmd_toolbox/vmd-interp.md](mmd_toolbox/vmd-interp.md) — 補間曲線評価
  - [mmd_toolbox/vmd-camera.md](mmd_toolbox/vmd-camera.md) — カメラ座標変換
- [shakevmd/shakevmd.md](shakevmd/shakevmd.md) — shakevmd

## 必要環境

- Python 3.11 以上

## 環境構築

リポジトリルートで:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## テスト実行

```sh
pytest            # 全テスト
pytest mmd_toolbox     # mmd_toolboxのみ
```

テストは外部サービス・ネットワーク・MMD本体を必要としない。

### MMD産テストデータについて

一部のテストはMMD本体で作成したVMDファイルを使用する。
未作成の場合、該当テストは
「MMD産テストデータがありません」というメッセージで失敗する。
作成手順(必要なファイルと条件)は
[mmd_toolbox/tests/data/README.md](mmd_toolbox/tests/data/README.md) を参照。
