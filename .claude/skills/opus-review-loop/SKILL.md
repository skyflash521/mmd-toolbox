---
name: opus-review-loop
description: opus-reviewer(Opusモデル、read-only)にレビューさせ、各指摘を実コードで検証して修正/反証/受容/保留に仕分け、再レビューを反復する。未解決ゼロ・千日手・要ユーザー判断のいずれかで終了。Opusモデルでコードレビューを回したいときに使う。
---

# Opus 反復レビュー・ループ

`opus-reviewer` サブエージェント(read-only, Opusモデル)にレビューさせ、Claude が各指摘を
実コードで検証して修正/反証/受容/保留に仕分け、再レビューさせる反復ループ。日本語で報告する。

レビュアーの起動契約と、1ラウンドを回して結末を出すまでの手順は
[review-loop-subagent](../review-loop-subagent/SKILL.md) スキル、レビュアーの正体に依らない
判断ロジックは [review-loop-judgement](../review-loop-judgement/SKILL.md) スキルを見よ。

このスキルが定めるのは、[`review-loop-subagent` が呼び出し元に委ねている項目](../review-loop-subagent/SKILL.md#呼び出し元が定める項目):

- **レビュアーエージェント**: `opus-reviewer`(`subagent_type: "opus-reviewer"`)
- **固定モデル**: `opus`。表示名は Opus(`Opus` / `claude-opus-*`)
