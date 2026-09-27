# vocal_analysis_cli 仕様書

層: **共有ドメイン層**([../../docs/conventions/layering.md §1](../../docs/conventions/layering.md#1-層タクソノミー))。

音声前段 [vocal_analysis](../vocal_analysis/vocal_analysis.md) を CLI から駆動するための共通引数群を定義し、検証と、
vocal_analysis へ渡す設定への解決までを担う。音声前段の実行そのものは持たない。

## 対象範囲

- 共通引数群は一括で登録する。一部の引数だけを非公開にする機構は持たない。

## 呼び出し側の要件

- 追加依存(`vocal-analysis` extra)を要する取り込みの失敗は、引数の検証をすべて終えた後・処理を開始する
  直前に、呼び出し側のツールが検査する。
- 実行デバイスの選択は、プロセス内で最初に GPU を照会するより前に適用する。CUDA は利用できるデバイスの
  集合を最初の照会で固定する。
