---
name: fable-review-loop
description: fable-reviewer(Fableモデル、read-only)にレビューさせ、各指摘を実コードで検証して修正/反証/受容/保留に仕分け、再レビューを反復する。未解決ゼロ・千日手・要ユーザー判断のいずれかで終了。Fableモデルでコードレビューを回したいときに使う。
---

# Fable 反復レビュー・ループ

`fable-reviewer` サブエージェント(read-only, Fableモデル)にレビューさせ、Claude が各指摘を
実コードで検証して修正/反証/受容/保留に仕分け、再レビューさせる反復ループ。日本語で報告する。

レビュアーの起動契約と、1ラウンドを回して結末を出すまでの手順は
[review-loop-subagent](../review-loop-subagent/SKILL.md) スキル、レビュアーの正体に依らない
判断ロジックは [review-loop-judgement](../review-loop-judgement/SKILL.md) スキルを見よ。

このスキルが定めるのは、[`review-loop-subagent` が呼び出し元に委ねている項目](../review-loop-subagent/SKILL.md#呼び出し元が定める項目):

- **レビュアーエージェント**: `fable-reviewer`(`subagent_type: "fable-reviewer"`)
- **固定モデル**: `fable`。表示名は Fable(`Fable` / `claude-fable-*`)
