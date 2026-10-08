from __future__ import annotations


PROTOCOL_VERSION = "phase3-2026-10-07-r5-subject-gated"

# One authoritative model-facing policy. Transport adapters must reference this
# constant rather than copying or extending it.
READ_SEARCH_POLICY = (
    "候选标题只是可能相关的旧记忆，不是事实正文。"
    "当前问题确实需要旧事时，先读最直接候选；没有合适候选再搜索。"
    "用户明确说自己以前告诉过、你应该知道或还记得某事时，"
    "必须在回答前读取直接候选，不能凭印象声称已经记得。"
    "普通情绪表达不等于要求翻查健康或其他保险柜。"
)

MODEL_RESPONSE_INSTRUCTION = (
    "像延续中的同一段对话一样直接回应用户。"
    + READ_SEARCH_POLICY
    + "不要向用户提候选目录、内部评分或工具过程，也不要使用 shell。"
)

CANDIDATE_INDEX_INSTRUCTION = READ_SEARCH_POLICY
SCOPE_CATALOG_INSTRUCTION = (
    "scope 只按目录说明选择；明确追忆只授权与当前问题直接对应的范围。"
)
