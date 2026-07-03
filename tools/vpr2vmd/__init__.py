"""vpr2vmd — vpr(VOCALOID プロジェクト)から口パク VMD を生成する独立 CLI(vpr2vmd.md)。

vpr の解析は vpr、口パクのキーフレーム生成は lipsync、VMD 出力は vmd に
委譲し、本パッケージは vpr 固有の入口処理(口形イベント確定・開き量)と CLI を担う。
"""

# 版の正本(vpr2vmd.md §4.2 の --version が表示する)。
__version__ = "0.0.1"
