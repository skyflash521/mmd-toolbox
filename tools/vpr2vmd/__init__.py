"""vpr2vmd — vpr(VOCALOID プロジェクト)からリップモーション VMD を生成する独立 CLI。

vpr の解析は vpr、リップモーションのキーフレーム生成は lipsync、VMD 出力は vmd に
委譲し、本パッケージは vpr 固有の入口処理(口形イベント確定・開き量)と CLI を担う。
"""

# 版の正本(--version が表示する)。
__version__ = "0.0.1"
