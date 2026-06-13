# CLAUDE.md

## プロジェクト概要

MikuMikuDance (MMD) 向けの雑多なユーティリティスクリプト群。ファイル変換、データ処理など、MMD関連の作業を補助するツールをまとめたリポジトリ。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `docs/specs/` | MMDファイル形式の詳細仕様(VMD/PMX)。バイナリレイアウトの正 |
| `mmd_toolbox/` | 共通ライブラリ。VMD読み書き・補間曲線評価・カメラ座標変換。仕様: `mmd_toolbox/mmd_toolbox.md` |
| `shakevmd/` | カメラVMDに手ぶれを焼き込むCLIツール。仕様: `shakevmd/shakevmd.md` |

各ツール・ライブラリの仕様書は、そのディレクトリ直下に置く(例: `shakevmd/shakevmd.md`)。
フォーマット層の機能(読み書き・MMD互換評価)はツール側に書かず mmd_toolbox に置く。

## 開発・テスト

- セットアップ: `pip install -e ".[dev]"`(ルートの `pyproject.toml`)
- テスト: ルートで `pytest`(全体)、`pytest mmd_toolbox` / `pytest shakevmd`(限定)。
  方針の詳細は `mmd_toolbox/mmd_toolbox.md` §4

## このファイルの編集ルール

このファイルを更新する際は単純に追記するのではなく、全体を再構成してから書き直すこと。
