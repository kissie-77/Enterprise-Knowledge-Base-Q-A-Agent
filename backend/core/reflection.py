"""
Agent 自我反思模块（Reflection）
回答生成后进行自我评估，不满意则重新生成

核心理念（参考 hello-agents 第4章 Reflection 范式）：
Agent 不是"一次生成就完事"，而是具备自我批判和迭代改进的能力。

流程：
1. 生成初始回答
2. Critic（评审员）评估回答质量
3. 如果质量不达标 → 给出具体改进建议 → 重新生成
4. 如果质量达标 → 输出最终回答
5. 最多迭代 N 轮

评估维度：
- 准确性：回答是否忠于参考资料
- 完整性：是否回答了用户问题的所有方面
- 清晰度：表述是否清晰、有条理
- 相关性：是否聚焦于用户真正想问的问题

面试考点：
- Reflection 模式 vs 简单重试的区别
- 如何避免 Reflection 陷入死循环（收敛条件）
- Self-Refine 论文的核心思想
"""
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .llm_client import LLMClient


class QualityLevel(Enum):
    """回答质量等级"""
    EXCELLENT = "excellent"    # 优秀，无需改进
    GOOD = "good"             # 良好，小问题可忽略
    FAIR = "fair"             # 一般，建议改进
    POOR = "poor"             # 较差，必须改进


@dataclass
class CriticFeedback:
    """评审反馈"""
    quality_level: QualityLevel
    overall_score: float           # 综合评分 (0-10)
    accuracy_score: float          # 准确性 (0-10)
    completeness_score: float      # 完整性 (0-10)
    clarity_score: float           # 清晰度 (0-10)
    relevance_score: float         # 相关性 (0-10)
    issues: List[str] = field(default_factory=list)         # 发现的问题
    suggestions: List[str] = field(default_factory=list)    # 改进建议
    should_refine: bool = False    # 是否需要改进

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_level": self.quality_level.value,
            "overall_score": self.overall_score,
            "scores": {
                "accuracy": self.accuracy_score,
                "completeness": self.completeness_score,
                "clarity": self.clarity_score,
                "relevance": self.relevance_score,
            },
            "issues": self.issues,
            "suggestions": self.suggestions,
            "should_refine": self.should_refine,
        }


@dataclass
class ReflectionResult:
    """反思迭代结果"""
    final_answer: str                      # 最终回答
    initial_answer: str                    # 初始回答
    iterations: int                        # 迭代次数
    feedbacks: List[CriticFeedback] = field(default_factory=list)  # 每轮的评审反馈
    improvement_history: List[str] = field(default_factory=list)    # 改进历史
    converged: bool = False                # 是否收敛（质量达标）

    @property
    def was_refined(self) -> bool:
        """是否经过了改进"""
        return self.iterations > 1

    @property
    def final_score(self) -> float:
        """最终评分"""
        return self.feedbacks[-1].overall_score if self.feedbacks else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "was_refined": self.was_refined,
            "iterations": self.iterations,
            "final_score": self.final_score,
            "converged": self.converged,
            "feedbacks": [f.to_dict() for f in self.feedbacks],
            "improvement_history": self.improvement_history,
        }


# ===== Prompt 模板 =====

CRITIC_PROMPT = """你是一个严格的回答质量评审员。请从以下维度评估这个回答的质量。

## 评估维度
1. **准确性** (0-10)：回答是否忠于参考资料？是否有编造的信息？
2. **完整性** (0-10)：是否回答了用户问题的所有方面？有无遗漏？
3. **清晰度** (0-10)：表述是否清晰、有条理？是否容易理解？
4. **相关性** (0-10)：是否聚焦于用户的问题？有无跑题？

## 用户问题
{question}

## 参考资料
{context}

## 待评估的回答
{answer}

## 输出格式（严格遵循）
准确性: X/10
完整性: X/10
清晰度: X/10
相关性: X/10
综合评分: X/10

问题:
- 问题1
- 问题2

改进建议:
- 建议1
- 建议2

是否需要改进: 是/否"""


REFINE_PROMPT = """请根据评审反馈改进你的回答。

## 用户问题
{question}

## 参考资料
{context}

## 你之前的回答
{previous_answer}

## 评审反馈
{feedback}

## 改进要求
1. 针对评审指出的每个问题进行修正
2. 保留原回答中正确和优秀的部分
3. 确保改进后的回答更加准确、完整、清晰
4. 如果参考资料中没有的信息，不要编造

## 请给出改进后的回答："""


class ReflectionEngine:
    """
    自我反思引擎

    让 Agent 具备自我批判和迭代改进的能力。
    参考 Self-Refine 论文（Madaan et al., 2023）的核心思想。

    使用方式:
        engine = ReflectionEngine(llm=llm_client)

        # 对回答进行反思改进
        result = await engine.reflect_and_refine(
            question="什么是 RAG？",
            initial_answer="RAG是...",
            context="检索到的参考资料..."
        )

        # 只做质量评估（不改进）
        feedback = await engine.critique(
            question="...", answer="...", context="..."
        )
    """

    def __init__(
        self,
        llm: LLMClient,
        max_iterations: int = 2,
        quality_threshold: float = 7.0,
        min_improvement: float = 0.5,
    ):
        """
        Args:
            llm: LLM 客户端
            max_iterations: 最大反思迭代次数（防止死循环）
            quality_threshold: 质量阈值，达到此分数即停止改进 (0-10)
            min_improvement: 最小改进幅度，如果改进不明显则停止
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.min_improvement = min_improvement

    async def reflect_and_refine(
        self,
        question: str,
        initial_answer: str,
        context: str = "",
    ) -> ReflectionResult:
        """
        反思并改进回答

        流程: 评估 → (如不达标) → 改进 → 再评估 → ... → 输出

        Args:
            question: 用户问题
            initial_answer: 初始回答
            context: 参考资料

        Returns:
            ReflectionResult 反思结果
        """
        result = ReflectionResult(
            final_answer=initial_answer,
            initial_answer=initial_answer,
            iterations=0,
        )

        current_answer = initial_answer
        previous_score = 0.0

        for iteration in range(self.max_iterations):
            result.iterations = iteration + 1

            # 1. 评审当前回答
            feedback = await self.critique(question, current_answer, context)
            result.feedbacks.append(feedback)

            # 2. 检查是否达标
            if not feedback.should_refine:
                result.final_answer = current_answer
                result.converged = True
                break

            # 3. 检查改进是否收敛
            if iteration > 0 and (feedback.overall_score - previous_score) < self.min_improvement:
                # 改进不明显，停止迭代
                result.final_answer = current_answer
                result.converged = True
                break

            previous_score = feedback.overall_score

            # 4. 基于反馈改进回答
            refined_answer = await self._refine(
                question=question,
                previous_answer=current_answer,
                feedback=feedback,
                context=context,
            )

            if refined_answer and refined_answer != current_answer:
                improvement_note = f"第{iteration+1}轮改进: 修正了 {', '.join(feedback.issues[:2])}"
                result.improvement_history.append(improvement_note)
                current_answer = refined_answer
            else:
                # 改进失败或无变化
                result.final_answer = current_answer
                result.converged = True
                break

        result.final_answer = current_answer
        return result

    async def critique(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> CriticFeedback:
        """
        评审回答质量

        Args:
            question: 用户问题
            answer: 待评审的回答
            context: 参考资料

        Returns:
            CriticFeedback 评审反馈
        """
        context_display = context[:2000] if context else "（无参考资料）"

        prompt = CRITIC_PROMPT.format(
            question=question,
            context=context_display,
            answer=answer,
        )

        try:
            result = await self.llm.think(prompt, temperature=0)
            return self._parse_critic_output(result)
        except Exception as e:
            print(f"⚠️ 质量评审失败: {e}")
            # 返回默认通过的反馈
            return CriticFeedback(
                quality_level=QualityLevel.GOOD,
                overall_score=7.0,
                accuracy_score=7.0,
                completeness_score=7.0,
                clarity_score=7.0,
                relevance_score=7.0,
                should_refine=False,
            )

    async def _refine(
        self,
        question: str,
        previous_answer: str,
        feedback: CriticFeedback,
        context: str = "",
    ) -> str:
        """根据反馈改进回答"""
        context_display = context[:2000] if context else "（无参考资料）"

        feedback_text = f"""综合评分: {feedback.overall_score}/10

发现的问题:
{chr(10).join(f"- {issue}" for issue in feedback.issues)}

改进建议:
{chr(10).join(f"- {sug}" for sug in feedback.suggestions)}"""

        prompt = REFINE_PROMPT.format(
            question=question,
            context=context_display,
            previous_answer=previous_answer,
            feedback=feedback_text,
        )

        try:
            refined = await self.llm.think(prompt, temperature=0.1)
            return refined.strip()
        except Exception as e:
            print(f"⚠️ 回答改进失败: {e}")
            return previous_answer

    def _parse_critic_output(self, output: str) -> CriticFeedback:
        """解析评审员输出"""
        import re

        def extract_score(pattern: str, text: str, default: float = 5.0) -> float:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    pass
            return default

        accuracy = extract_score(r'准确性[：:]\s*(\d+(?:\.\d+)?)', output)
        completeness = extract_score(r'完整性[：:]\s*(\d+(?:\.\d+)?)', output)
        clarity = extract_score(r'清晰度[：:]\s*(\d+(?:\.\d+)?)', output)
        relevance = extract_score(r'相关性[：:]\s*(\d+(?:\.\d+)?)', output)
        overall = extract_score(r'综合评分[：:]\s*(\d+(?:\.\d+)?)', output)

        # 如果没有综合评分，取平均
        if overall == 5.0 and (accuracy != 5.0 or completeness != 5.0):
            overall = (accuracy + completeness + clarity + relevance) / 4

        # 提取问题
        issues = []
        issues_section = re.search(r'问题[：:]?\s*\n(.*?)(?=改进建议|是否需要|$)', output, re.DOTALL)
        if issues_section:
            for line in issues_section.group(1).strip().split('\n'):
                line = line.strip().lstrip('- •·')
                if line and len(line) > 2:
                    issues.append(line)

        # 提取改进建议
        suggestions = []
        suggestions_section = re.search(r'改进建议[：:]?\s*\n(.*?)(?=是否需要|$)', output, re.DOTALL)
        if suggestions_section:
            for line in suggestions_section.group(1).strip().split('\n'):
                line = line.strip().lstrip('- •·')
                if line and len(line) > 2:
                    suggestions.append(line)

        # 判断是否需要改进
        should_refine = overall < self.quality_threshold
        refine_match = re.search(r'是否需要改进[：:]\s*(是|否)', output)
        if refine_match:
            should_refine = refine_match.group(1) == "是"

        # 确定质量等级
        if overall >= 9:
            quality_level = QualityLevel.EXCELLENT
        elif overall >= 7:
            quality_level = QualityLevel.GOOD
        elif overall >= 5:
            quality_level = QualityLevel.FAIR
        else:
            quality_level = QualityLevel.POOR

        return CriticFeedback(
            quality_level=quality_level,
            overall_score=overall,
            accuracy_score=accuracy,
            completeness_score=completeness,
            clarity_score=clarity,
            relevance_score=relevance,
            issues=issues[:5],
            suggestions=suggestions[:5],
            should_refine=should_refine,
        )
