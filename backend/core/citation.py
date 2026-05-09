"""
回答可溯源模块（Citation & Attribution）
让 Agent 的每句回答都能追溯到知识库中的原始来源

核心功能：
1. 生成时引用：在 LLM 生成回答时就要求标注引用编号
2. 事后归因：回答生成后，自动将句子与检索片段对齐
3. 引用验证：检验回答内容是否确实来自引用的片段（防幻觉）
4. 置信度评估：对引用标注的准确性给出置信度

面试考点：
- 为什么 RAG 系统需要 Citation？（可信度 + 可审计 + 防幻觉）
- Inline Citation vs Post-hoc Attribution 的优劣
- 如何检测 LLM 的幻觉（Hallucination Detection）
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .llm_client import LLMClient


@dataclass
class CitationSpan:
    """单条引用标注"""
    text: str                              # 被引用的回答文本片段
    source_index: int                      # 引用来源的索引编号（对应检索结果）
    source_content: str = ""               # 来源原文（片段）
    source_file: str = ""                  # 来源文件名
    confidence: float = 0.0                # 引用置信度 (0-1)
    is_verified: bool = False              # 是否经过验证

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_index": self.source_index,
            "source_content": self.source_content[:200],
            "source_file": self.source_file,
            "confidence": round(self.confidence, 2),
            "is_verified": self.is_verified,
        }


@dataclass
class CitedAnswer:
    """带引用的回答"""
    raw_answer: str                        # 原始回答（带引用标记）
    clean_answer: str                      # 清洁回答（去掉引用标记的纯文本）
    formatted_answer: str                  # 格式化回答（引用以上标形式展示）
    citations: List[CitationSpan] = field(default_factory=list)  # 引用列表
    sources: List[Dict[str, Any]] = field(default_factory=list)  # 来源列表
    hallucination_risk: float = 0.0        # 幻觉风险评分 (0-1, 越高越可能有幻觉)
    coverage_score: float = 0.0            # 来源覆盖度 (0-1, 回答中有多少内容有来源支撑)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.formatted_answer,
            "clean_answer": self.clean_answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "hallucination_risk": round(self.hallucination_risk, 2),
            "coverage_score": round(self.coverage_score, 2),
            "citation_count": len(self.citations),
        }


# ===== Prompt 模板 =====

CITATION_GENERATION_PROMPT = """你是一个专业的知识库问答助手。请基于提供的参考资料回答用户问题。

## 重要要求
1. 回答必须基于参考资料中的内容
2. 每个关键信息点后面用 [数字] 标注引用来源，如 [1]、[2]
3. 如果参考资料中没有相关信息，明确说明"根据现有资料未找到相关信息"
4. 不要编造参考资料中不存在的信息

## 参考资料
{context}

## 用户问题
{question}

## 请给出带引用标注的回答："""


ATTRIBUTION_PROMPT = """请判断以下回答中的每句话是否能从参考资料中找到依据。

## 参考资料
{context}

## 回答内容
{answer}

## 任务
对回答中的每句话进行溯源，输出格式如下（每行一条）：
句子内容 | 来源编号(1-{num_sources}) | 置信度(高/中/低/无来源)

如果某句话在参考资料中找不到依据，来源编号写 0，置信度写"无来源"。

## 溯源结果："""


HALLUCINATION_CHECK_PROMPT = """请检查以下回答是否存在"幻觉"（即回答中包含参考资料中不存在的信息）。

## 参考资料
{context}

## 回答
{answer}

## 请分析
1. 回答中有哪些信息在参考资料中有明确依据？（列出）
2. 回答中有哪些信息在参考资料中找不到依据？（列出）
3. 给出整体幻觉风险评分（0-10分，0=无幻觉，10=完全幻觉）

## 格式要求
最后一行必须是: 幻觉风险评分: X/10

## 分析："""


class CitationEngine:
    """
    引用溯源引擎

    两种工作模式：
    1. 生成时引用（Inline Citation）：生成回答时就要求带引用标注
    2. 事后归因（Post-hoc Attribution）：生成回答后自动归因

    使用方式:
        engine = CitationEngine(llm=llm_client)

        # 模式1: 生成带引用的回答
        cited = await engine.generate_with_citations(
            question="什么是RAG？",
            contexts=[{"content": "RAG是...", "source": "doc1.pdf"}]
        )

        # 模式2: 对已有回答进行事后归因
        cited = await engine.attribute_answer(
            answer="RAG是一种检索增强技术...",
            contexts=[...]
        )

        # 模式3: 幻觉检测
        risk = await engine.check_hallucination(answer, contexts)
    """

    def __init__(self, llm: LLMClient):
        """
        Args:
            llm: LLM 客户端
        """
        self.llm = llm

    async def generate_with_citations(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
    ) -> CitedAnswer:
        """
        生成带引用标注的回答（Inline Citation 模式）

        Args:
            question: 用户问题
            contexts: 检索结果列表，每项需包含 content 和 metadata

        Returns:
            CitedAnswer 带引用的回答
        """
        if not contexts:
            return CitedAnswer(
                raw_answer="根据现有知识库，未找到相关信息。",
                clean_answer="根据现有知识库，未找到相关信息。",
                formatted_answer="根据现有知识库，未找到相关信息。",
                hallucination_risk=0.0,
                coverage_score=0.0,
            )

        # 构建带编号的上下文
        context_text = self._format_contexts_with_numbers(contexts)

        # 生成带引用的回答
        prompt = CITATION_GENERATION_PROMPT.format(
            context=context_text,
            question=question,
        )

        raw_answer = await self.llm.think(prompt, temperature=0)

        # 解析引用标注
        citations = self._extract_citations(raw_answer, contexts)

        # 生成格式化版本
        clean_answer = self._remove_citation_markers(raw_answer)
        formatted_answer = self._format_citations_display(raw_answer, contexts)

        # 计算覆盖度
        coverage = self._calculate_coverage(raw_answer, citations)

        # 构建来源列表
        sources = self._build_sources_list(contexts, citations)

        return CitedAnswer(
            raw_answer=raw_answer,
            clean_answer=clean_answer,
            formatted_answer=formatted_answer,
            citations=citations,
            sources=sources,
            coverage_score=coverage,
        )

    async def attribute_answer(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> CitedAnswer:
        """
        事后归因：对已有回答进行来源归因

        Args:
            answer: 已生成的回答
            contexts: 检索结果

        Returns:
            CitedAnswer 归因后的回答
        """
        if not contexts:
            return CitedAnswer(
                raw_answer=answer,
                clean_answer=answer,
                formatted_answer=answer,
            )

        context_text = self._format_contexts_with_numbers(contexts)

        prompt = ATTRIBUTION_PROMPT.format(
            context=context_text,
            answer=answer,
            num_sources=len(contexts),
        )

        try:
            result = await self.llm.think(prompt, temperature=0)
            citations = self._parse_attribution_result(result, contexts)

            # 在回答中插入引用标注
            formatted_answer = self._insert_citations_into_answer(answer, citations)
            coverage = self._calculate_coverage_from_attribution(citations)

            sources = self._build_sources_list(contexts, citations)

            return CitedAnswer(
                raw_answer=answer,
                clean_answer=answer,
                formatted_answer=formatted_answer,
                citations=citations,
                sources=sources,
                coverage_score=coverage,
            )
        except Exception as e:
            print(f"⚠️ 事后归因失败: {e}")
            return CitedAnswer(
                raw_answer=answer,
                clean_answer=answer,
                formatted_answer=answer,
            )

    async def check_hallucination(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
    ) -> float:
        """
        幻觉检测：检查回答是否包含知识库中不存在的信息

        Args:
            answer: 回答文本
            contexts: 检索上下文

        Returns:
            幻觉风险评分 (0-1)
        """
        if not contexts or not answer:
            return 0.5  # 无法判断

        context_text = self._format_contexts_with_numbers(contexts)

        prompt = HALLUCINATION_CHECK_PROMPT.format(
            context=context_text,
            answer=answer,
        )

        try:
            result = await self.llm.think(prompt, temperature=0)

            # 提取评分
            score_match = re.search(r'幻觉风险评分[：:]\s*(\d+)\s*/\s*10', result)
            if score_match:
                score = int(score_match.group(1))
                return min(score / 10.0, 1.0)

            # 备用解析
            score_match = re.search(r'(\d+)\s*/\s*10', result)
            if score_match:
                score = int(score_match.group(1))
                return min(score / 10.0, 1.0)

            return 0.5  # 无法解析时返回中等风险
        except Exception as e:
            print(f"⚠️ 幻觉检测失败: {e}")
            return 0.5

    # ===== 内部方法 =====

    def _format_contexts_with_numbers(self, contexts: List[Dict[str, Any]]) -> str:
        """将检索结果格式化为带编号的文本"""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            content = ctx.get("content", "")
            source = ctx.get("metadata", {}).get("source", ctx.get("source", "未知来源"))
            parts.append(f"[{i}] (来源: {source})\n{content}")
        return "\n\n".join(parts)

    def _extract_citations(
        self, answer: str, contexts: List[Dict[str, Any]]
    ) -> List[CitationSpan]:
        """从带标注的回答中提取引用信息"""
        citations = []

        # 匹配 [数字] 模式
        pattern = r'([^[]+?)\[(\d+)\]'
        matches = re.finditer(pattern, answer)

        for match in matches:
            text = match.group(1).strip()
            source_idx = int(match.group(2)) - 1  # 转为0-based索引

            if 0 <= source_idx < len(contexts):
                ctx = contexts[source_idx]
                citation = CitationSpan(
                    text=text,
                    source_index=source_idx + 1,
                    source_content=ctx.get("content", "")[:200],
                    source_file=ctx.get("metadata", {}).get("source", ""),
                    confidence=0.8,  # 默认较高置信度（LLM 主动标注的）
                    is_verified=False,
                )
                citations.append(citation)

        return citations

    def _remove_citation_markers(self, text: str) -> str:
        """移除引用标记，得到纯净文本"""
        return re.sub(r'\[\d+\]', '', text).strip()

    def _format_citations_display(
        self, answer: str, contexts: List[Dict[str, Any]]
    ) -> str:
        """格式化引用展示（保留引用编号 + 底部附来源列表）"""
        # 保持正文中的引用编号
        formatted = answer

        # 添加来源注释
        sources_used = set(map(int, re.findall(r'\[(\d+)\]', answer)))
        if sources_used:
            formatted += "\n\n---\n**参考来源：**\n"
            for idx in sorted(sources_used):
                if 0 < idx <= len(contexts):
                    ctx = contexts[idx - 1]
                    source = ctx.get("metadata", {}).get("source", ctx.get("source", "未知"))
                    formatted += f"- [{idx}] {source}\n"

        return formatted

    def _parse_attribution_result(
        self, result: str, contexts: List[Dict[str, Any]]
    ) -> List[CitationSpan]:
        """解析事后归因的 LLM 输出"""
        citations = []

        for line in result.strip().split('\n'):
            line = line.strip()
            if '|' not in line:
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                continue

            text = parts[0]
            try:
                source_idx = int(re.search(r'\d+', parts[1]).group())
            except (AttributeError, ValueError):
                source_idx = 0

            confidence_text = parts[2] if len(parts) > 2 else ""

            # 转换置信度
            confidence_map = {"高": 0.9, "中": 0.6, "低": 0.3, "无来源": 0.0}
            confidence = confidence_map.get(confidence_text, 0.5)

            if source_idx > 0 and source_idx <= len(contexts):
                ctx = contexts[source_idx - 1]
                citations.append(CitationSpan(
                    text=text,
                    source_index=source_idx,
                    source_content=ctx.get("content", "")[:200],
                    source_file=ctx.get("metadata", {}).get("source", ""),
                    confidence=confidence,
                    is_verified=True,
                ))
            elif text:
                # 无来源的句子也记录
                citations.append(CitationSpan(
                    text=text,
                    source_index=0,
                    confidence=0.0,
                    is_verified=True,
                ))

        return citations

    def _insert_citations_into_answer(
        self, answer: str, citations: List[CitationSpan]
    ) -> str:
        """将归因结果以引用标注形式插入回答"""
        # 简单策略：在每个有来源的句子末尾加上引用编号
        formatted = answer
        for citation in reversed(citations):  # 倒序避免位置偏移
            if citation.source_index > 0 and citation.text:
                # 尝试在回答中找到对应文本并添加标注
                if citation.text in formatted:
                    formatted = formatted.replace(
                        citation.text,
                        f"{citation.text}[{citation.source_index}]",
                        1,
                    )
        return formatted

    def _calculate_coverage(self, answer: str, citations: List[CitationSpan]) -> float:
        """计算引用覆盖度"""
        if not answer:
            return 0.0

        total_length = len(self._remove_citation_markers(answer))
        if total_length == 0:
            return 0.0

        cited_length = sum(len(c.text) for c in citations if c.source_index > 0)
        return min(cited_length / total_length, 1.0)

    def _calculate_coverage_from_attribution(self, citations: List[CitationSpan]) -> float:
        """从归因结果计算覆盖度"""
        if not citations:
            return 0.0

        total = len(citations)
        sourced = sum(1 for c in citations if c.source_index > 0 and c.confidence > 0.3)
        return sourced / total

    def _build_sources_list(
        self, contexts: List[Dict[str, Any]], citations: List[CitationSpan]
    ) -> List[Dict[str, Any]]:
        """构建去重的来源列表"""
        used_indices = set(c.source_index for c in citations if c.source_index > 0)
        sources = []

        for idx in sorted(used_indices):
            if 0 < idx <= len(contexts):
                ctx = contexts[idx - 1]
                sources.append({
                    "index": idx,
                    "source": ctx.get("metadata", {}).get("source", ctx.get("source", "未知")),
                    "content_preview": ctx.get("content", "")[:150],
                    "citation_count": sum(1 for c in citations if c.source_index == idx),
                })

        return sources
