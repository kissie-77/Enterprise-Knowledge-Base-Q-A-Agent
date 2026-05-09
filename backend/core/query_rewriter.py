"""
查询改写模块（Query Rewriting）
用 LLM 自动优化用户的模糊提问，提升 RAG 检索召回率

核心策略：
1. 意图澄清：将模糊问题转化为明确的检索查询
2. 多查询生成：一个问题生成多个角度的子查询
3. 历史感知改写：结合对话上下文，补全指代和省略
4. HyDE（假设性文档嵌入）：先生成假设性答案，再用答案去检索

面试考点：
- 为什么需要查询改写？（用户提问 ≠ 好的检索 query）
- 多查询生成 vs 查询扩展 的区别
- HyDE 的原理和适用场景
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .llm_client import LLMClient


@dataclass
class RewriteResult:
    """查询改写结果"""
    original_query: str                    # 原始查询
    rewritten_query: str                   # 主改写查询
    sub_queries: List[str] = field(default_factory=list)  # 子查询列表
    hyde_passage: str = ""                 # HyDE 假设性文档
    strategy_used: str = ""                # 使用的改写策略
    context_used: bool = False             # 是否使用了对话上下文

    def get_all_queries(self) -> List[str]:
        """获取所有用于检索的查询（去重）"""
        queries = [self.rewritten_query]
        queries.extend(self.sub_queries)
        if self.hyde_passage:
            queries.append(self.hyde_passage)
        # 去重保序
        seen = set()
        result = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                result.append(q)
        return result


# ===== Prompt 模板 =====

REWRITE_PROMPT = """你是一个查询优化专家。请将用户的问题改写为更适合在知识库中检索的查询。

## 改写原则
1. 去掉口语化表达，保留核心关键词
2. 补充可能的同义词或相关术语
3. 如果问题模糊，推测用户最可能的意图
4. 改写后的查询应该简洁、精确

## 用户原始问题
{query}

{context_section}

## 输出要求
请直接输出改写后的查询（一行文字，不需要解释）："""


MULTI_QUERY_PROMPT = """你是一个检索查询生成专家。请将用户问题从不同角度拆解为 {num_queries} 个子查询，以便从知识库中检索到更全面的信息。

## 用户问题
{query}

{context_section}

## 要求
1. 每个子查询关注问题的不同方面或使用不同的表述方式
2. 子查询应该简洁，适合用于向量检索
3. 不要重复，每个查询要有独特的角度

## 输出格式
每行一个子查询，不需要编号或前缀："""


HYDE_PROMPT = """请针对以下问题，写一段可能出现在知识库文档中的回答段落。
这段文字应该是"如果知识库中有相关内容，它大概会是什么样子"。

## 问题
{query}

## 要求
1. 写一段 100-200 字的假设性文档段落
2. 使用专业、正式的语言风格
3. 包含可能的关键术语和核心概念
4. 不需要完全准确，但要像是一篇真实文档的片段

## 假设性文档段落："""


CONTEXT_AWARE_REWRITE_PROMPT = """你是一个对话理解专家。请结合对话历史，将用户最新的问题改写为一个独立、完整的查询。

## 对话历史
{history}

## 用户最新问题
{query}

## 改写要求
1. 补全用户问题中的指代词（"它"、"这个"、"上面说的"等）
2. 补全省略的主语或对象
3. 改写后的查询应该脱离上下文也能理解
4. 保持简洁，适合检索

## 改写后的完整查询（一行文字）："""


class QueryRewriter:
    """
    查询改写器

    将用户的原始提问优化为更适合检索的查询形式，
    显著提升 RAG 系统的检索召回率和准确率。

    使用方式:
        rewriter = QueryRewriter(llm=llm_client)

        # 基础改写
        result = await rewriter.rewrite("这个东西怎么用")

        # 多查询生成
        result = await rewriter.rewrite("机器学习的应用", strategy="multi_query")

        # 带上下文的改写（多轮对话场景）
        result = await rewriter.rewrite(
            "它的优缺点是什么？",
            conversation_history="用户: 什么是RAG？\n助手: RAG是检索增强生成..."
        )

        # HyDE 策略
        result = await rewriter.rewrite("如何提升模型性能", strategy="hyde")
    """

    def __init__(
        self,
        llm: LLMClient,
        default_strategy: str = "auto",
        num_sub_queries: int = 3,
        enable_hyde: bool = True,
    ):
        """
        初始化查询改写器

        Args:
            llm: LLM 客户端
            default_strategy: 默认改写策略 (auto/simple/multi_query/hyde/context_aware)
            num_sub_queries: 多查询模式下生成的子查询数量
            enable_hyde: 是否启用 HyDE 策略
        """
        self.llm = llm
        self.default_strategy = default_strategy
        self.num_sub_queries = num_sub_queries
        self.enable_hyde = enable_hyde

    async def rewrite(
        self,
        query: str,
        strategy: Optional[str] = None,
        conversation_history: str = "",
        **kwargs,
    ) -> RewriteResult:
        """
        执行查询改写

        Args:
            query: 用户原始查询
            strategy: 改写策略（覆盖默认值）
                - "auto": 自动选择最佳策略
                - "simple": 简单改写（去口语化 + 关键词优化）
                - "multi_query": 多角度子查询生成
                - "hyde": HyDE 假设性文档嵌入
                - "context_aware": 上下文感知改写（多轮对话）
                - "full": 完整流水线（所有策略组合）
            conversation_history: 对话历史（用于上下文感知改写）

        Returns:
            RewriteResult 改写结果
        """
        strategy = strategy or self.default_strategy

        # 自动策略选择
        if strategy == "auto":
            strategy = self._auto_select_strategy(query, conversation_history)

        result = RewriteResult(
            original_query=query,
            rewritten_query=query,  # 默认值
            strategy_used=strategy,
        )

        if strategy == "simple":
            result.rewritten_query = await self._simple_rewrite(query, conversation_history)

        elif strategy == "multi_query":
            result.rewritten_query = await self._simple_rewrite(query, conversation_history)
            result.sub_queries = await self._generate_multi_queries(query, conversation_history)

        elif strategy == "hyde":
            result.rewritten_query = await self._simple_rewrite(query, conversation_history)
            result.hyde_passage = await self._generate_hyde(query)

        elif strategy == "context_aware":
            result.rewritten_query = await self._context_aware_rewrite(query, conversation_history)
            result.context_used = True

        elif strategy == "full":
            # 完整流水线：上下文改写 → 多查询 → HyDE
            if conversation_history:
                result.rewritten_query = await self._context_aware_rewrite(query, conversation_history)
                result.context_used = True
            else:
                result.rewritten_query = await self._simple_rewrite(query, conversation_history)

            result.sub_queries = await self._generate_multi_queries(
                result.rewritten_query, ""
            )

            if self.enable_hyde:
                result.hyde_passage = await self._generate_hyde(result.rewritten_query)

        else:
            # 未知策略，回退到简单改写
            result.rewritten_query = await self._simple_rewrite(query, conversation_history)

        return result

    def _auto_select_strategy(self, query: str, conversation_history: str) -> str:
        """
        自动选择最佳改写策略

        决策逻辑：
        - 有对话历史 + 短问题/有指代 → context_aware
        - 复杂长问题 → multi_query
        - 简单短问题 → simple
        """
        # 检测是否有指代词或省略
        has_reference = any(word in query for word in [
            "它", "这个", "那个", "上面", "刚才", "之前说的",
            "这", "那", "他们", "其", "该"
        ])

        if conversation_history and (has_reference or len(query) < 10):
            return "context_aware"
        elif len(query) > 30 or "和" in query or "以及" in query or "？" in query:
            return "multi_query"
        else:
            return "simple"

    async def _simple_rewrite(self, query: str, conversation_history: str = "") -> str:
        """简单改写：去口语化 + 关键词优化"""
        context_section = ""
        if conversation_history:
            context_section = f"## 对话上下文（供参考）\n{conversation_history[-500:]}"

        prompt = REWRITE_PROMPT.format(
            query=query,
            context_section=context_section,
        )

        try:
            result = await self.llm.think(prompt, temperature=0)
            # 清理结果
            result = result.strip().strip('"').strip("'")
            # 如果改写结果为空或太短，返回原始查询
            if not result or len(result) < 3:
                return query
            return result
        except Exception as e:
            print(f"⚠️ 查询改写失败: {e}")
            return query

    async def _generate_multi_queries(
        self, query: str, conversation_history: str = ""
    ) -> List[str]:
        """多查询生成：从不同角度拆解问题"""
        context_section = ""
        if conversation_history:
            context_section = f"## 对话上下文\n{conversation_history[-300:]}"

        prompt = MULTI_QUERY_PROMPT.format(
            query=query,
            num_queries=self.num_sub_queries,
            context_section=context_section,
        )

        try:
            result = await self.llm.think(prompt, temperature=0.3)
            # 解析多行输出
            lines = [
                line.strip().lstrip("0123456789.-) ").strip()
                for line in result.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            # 去重并限制数量
            seen = {query}  # 不包含原始查询
            sub_queries = []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    sub_queries.append(line)
            return sub_queries[:self.num_sub_queries]
        except Exception as e:
            print(f"⚠️ 多查询生成失败: {e}")
            return []

    async def _generate_hyde(self, query: str) -> str:
        """
        HyDE（Hypothetical Document Embedding）
        生成假设性文档片段，用其嵌入向量去检索真实文档

        原理：假设性答案与真实答案在向量空间中距离更近
        """
        prompt = HYDE_PROMPT.format(query=query)

        try:
            result = await self.llm.think(prompt, temperature=0.5)
            result = result.strip()
            # 限制长度
            if len(result) > 500:
                result = result[:500]
            return result
        except Exception as e:
            print(f"⚠️ HyDE 生成失败: {e}")
            return ""

    async def _context_aware_rewrite(self, query: str, conversation_history: str) -> str:
        """
        上下文感知改写
        结合对话历史，将包含指代/省略的问题改写为完整独立的查询
        """
        if not conversation_history:
            return await self._simple_rewrite(query)

        prompt = CONTEXT_AWARE_REWRITE_PROMPT.format(
            history=conversation_history[-800:],
            query=query,
        )

        try:
            result = await self.llm.think(prompt, temperature=0)
            result = result.strip().strip('"').strip("'")
            if not result or len(result) < 3:
                return query
            return result
        except Exception as e:
            print(f"⚠️ 上下文感知改写失败: {e}")
            return query

    def get_strategy_description(self, strategy: str) -> str:
        """获取策略的中文描述"""
        descriptions = {
            "auto": "自动选择",
            "simple": "简单改写（去口语化）",
            "multi_query": "多角度子查询生成",
            "hyde": "HyDE 假设性文档嵌入",
            "context_aware": "上下文感知改写",
            "full": "完整流水线",
        }
        return descriptions.get(strategy, strategy)
