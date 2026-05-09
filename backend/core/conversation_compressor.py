"""
对话摘要压缩模块（Conversation Compression）
长对话自动摘要，避免 token 爆炸，保留关键信息

核心问题：
- LLM 的上下文窗口有限（4K/8K/32K/128K）
- 多轮对话会快速消耗 token 配额
- 简单截断会丢失早期重要信息

解决方案（渐进式压缩策略）：
1. 滑动窗口：最近 N 轮保持原文
2. 摘要压缩：超出窗口的旧对话，用 LLM 生成摘要
3. 关键信息提取：识别并保留重要实体、结论、用户偏好
4. 分层压缩：越旧的对话压缩比越高

面试考点：
- 为什么不能简单用 "取最近N条" 解决？
- 压缩 vs 截断 的信息保留率对比
- 如何平衡压缩质量和额外 LLM 调用成本
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .llm_client import LLMClient


@dataclass
class CompressedContext:
    """压缩后的上下文"""
    summary: str                           # 对话摘要
    key_facts: List[str]                   # 关键信息点
    recent_messages: List[Dict[str, str]]  # 最近保留的原始消息
    total_original_tokens: int = 0         # 原始总 token 数（估算）
    compressed_tokens: int = 0             # 压缩后 token 数（估算）
    compression_ratio: float = 0.0         # 压缩比

    def get_context_messages(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        生成可直接用于 LLM 调用的消息列表

        格式:
        [system_prompt] + [摘要作为system补充] + [最近消息]
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 将摘要和关键信息作为上下文注入
        if self.summary or self.key_facts:
            context_parts = []
            if self.summary:
                context_parts.append(f"对话摘要：{self.summary}")
            if self.key_facts:
                facts_text = "\n".join(f"- {fact}" for fact in self.key_facts)
                context_parts.append(f"关键信息：\n{facts_text}")

            messages.append({
                "role": "system",
                "content": "以下是之前对话的摘要和关键信息：\n\n" + "\n\n".join(context_parts),
            })

        # 最近的原始消息
        messages.extend(self.recent_messages)

        return messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "key_facts": self.key_facts,
            "recent_messages_count": len(self.recent_messages),
            "total_original_tokens": self.total_original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 2),
        }


# ===== Prompt 模板 =====

SUMMARIZE_PROMPT = """请将以下对话历史压缩为一段简洁的摘要，保留关键信息。

## 压缩要求
1. 保留用户的核心问题和意图
2. 保留助手给出的关键结论和答案
3. 保留重要的事实性信息（数字、名称、日期等）
4. 去掉寒暄、重复、过渡语句
5. 摘要长度不超过 200 字

## 对话历史
{conversation}

## 压缩摘要："""


KEY_FACTS_PROMPT = """请从以下对话中提取关键信息点。这些信息在后续对话中可能被引用。

## 对话内容
{conversation}

## 提取要求
1. 提取具体的事实性信息（结论、数据、决策等）
2. 提取用户表达的偏好或需求
3. 提取已经确认的重要信息
4. 每条信息点一行，最多 8 条
5. 不要提取过于笼统的信息

## 关键信息点（每行一条）："""


INCREMENTAL_SUMMARY_PROMPT = """请将已有摘要和新的对话内容合并为一段更新后的摘要。

## 已有摘要
{existing_summary}

## 新的对话内容
{new_conversation}

## 合并要求
1. 保留已有摘要中仍然相关的信息
2. 整合新对话中的关键信息
3. 如果新对话推翻了旧信息，以新信息为准
4. 更新后的摘要不超过 300 字

## 更新后的摘要："""


class ConversationCompressor:
    """
    对话压缩器

    当对话轮数超过窗口大小时，自动将旧对话压缩为摘要，
    既节省 token 又保留关键信息。

    使用方式:
        compressor = ConversationCompressor(llm=llm_client, window_size=6)

        # 每次对话后检查是否需要压缩
        compressed = await compressor.compress_if_needed(messages)

        # 获取压缩后的上下文用于 LLM 调用
        context_messages = compressed.get_context_messages(system_prompt="...")

        # 强制压缩
        compressed = await compressor.compress(messages)
    """

    def __init__(
        self,
        llm: LLMClient,
        window_size: int = 6,
        max_tokens_estimate: int = 4000,
        chars_per_token: int = 2,
    ):
        """
        Args:
            llm: LLM 客户端
            window_size: 滑动窗口大小（保留最近N轮对话原文）
            max_tokens_estimate: 触发压缩的估算 token 阈值
            chars_per_token: 字符/token 换算比（中文约2，英文约4）
        """
        self.llm = llm
        self.window_size = window_size
        self.max_tokens_estimate = max_tokens_estimate
        self.chars_per_token = chars_per_token

        # 维护的摘要状态
        self._current_summary: str = ""
        self._key_facts: List[str] = []
        self._compression_count: int = 0

    async def compress_if_needed(
        self,
        messages: List[Dict[str, str]],
    ) -> CompressedContext:
        """
        检查并按需压缩对话

        Args:
            messages: 完整的对话消息列表（不含 system prompt）

        Returns:
            CompressedContext 压缩后的上下文
        """
        total_chars = sum(len(m.get("content", "")) for m in messages)
        total_tokens_est = total_chars // self.chars_per_token
        num_turns = len(messages) // 2  # 一轮 = user + assistant

        # 判断是否需要压缩
        needs_compression = (
            num_turns > self.window_size or
            total_tokens_est > self.max_tokens_estimate
        )

        if not needs_compression:
            # 不需要压缩，直接返回
            return CompressedContext(
                summary=self._current_summary,
                key_facts=self._key_facts,
                recent_messages=messages,
                total_original_tokens=total_tokens_est,
                compressed_tokens=total_tokens_est,
                compression_ratio=1.0,
            )

        # 需要压缩
        return await self.compress(messages)

    async def compress(
        self,
        messages: List[Dict[str, str]],
    ) -> CompressedContext:
        """
        执行对话压缩

        策略：
        - 最近 window_size 轮保持原文
        - 之前的对话生成摘要 + 关键信息提取

        Args:
            messages: 完整的对话消息列表

        Returns:
            CompressedContext
        """
        # 分割：旧消息（需要压缩）和新消息（保留原文）
        window_messages_count = self.window_size * 2  # 每轮2条消息
        if len(messages) <= window_messages_count:
            # 不需要压缩
            return CompressedContext(
                summary=self._current_summary,
                key_facts=self._key_facts,
                recent_messages=messages,
                total_original_tokens=self._estimate_tokens(messages),
                compressed_tokens=self._estimate_tokens(messages),
                compression_ratio=1.0,
            )

        old_messages = messages[:-window_messages_count]
        recent_messages = messages[-window_messages_count:]

        # 增量压缩：将旧消息的摘要与现有摘要合并
        old_conversation = self._format_messages(old_messages)

        if self._current_summary:
            # 增量更新摘要
            new_summary = await self._incremental_summarize(
                existing_summary=self._current_summary,
                new_conversation=old_conversation,
            )
        else:
            # 首次压缩
            new_summary = await self._summarize(old_conversation)

        # 提取关键信息
        new_key_facts = await self._extract_key_facts(old_conversation)

        # 合并关键信息（去重、限制数量）
        merged_facts = self._merge_key_facts(self._key_facts, new_key_facts)

        # 更新状态
        self._current_summary = new_summary
        self._key_facts = merged_facts
        self._compression_count += 1

        # 计算压缩统计
        original_tokens = self._estimate_tokens(messages)
        summary_tokens = len(new_summary) // self.chars_per_token
        facts_tokens = sum(len(f) for f in merged_facts) // self.chars_per_token
        recent_tokens = self._estimate_tokens(recent_messages)
        compressed_tokens = summary_tokens + facts_tokens + recent_tokens

        return CompressedContext(
            summary=new_summary,
            key_facts=merged_facts,
            recent_messages=recent_messages,
            total_original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
        )

    async def _summarize(self, conversation: str) -> str:
        """生成对话摘要"""
        prompt = SUMMARIZE_PROMPT.format(conversation=conversation[:3000])

        try:
            result = await self.llm.think(prompt, temperature=0)
            return result.strip()
        except Exception as e:
            print(f"⚠️ 摘要生成失败: {e}")
            # 降级：简单截断
            return conversation[:300] + "..."

    async def _incremental_summarize(
        self, existing_summary: str, new_conversation: str
    ) -> str:
        """增量更新摘要"""
        prompt = INCREMENTAL_SUMMARY_PROMPT.format(
            existing_summary=existing_summary,
            new_conversation=new_conversation[:2000],
        )

        try:
            result = await self.llm.think(prompt, temperature=0)
            return result.strip()
        except Exception as e:
            print(f"⚠️ 增量摘要失败: {e}")
            # 降级：拼接
            return f"{existing_summary}\n\n补充: {new_conversation[:200]}"

    async def _extract_key_facts(self, conversation: str) -> List[str]:
        """提取关键信息点"""
        prompt = KEY_FACTS_PROMPT.format(conversation=conversation[:2000])

        try:
            result = await self.llm.think(prompt, temperature=0)
            facts = []
            for line in result.strip().split('\n'):
                line = line.strip().lstrip('- •·0123456789.)')
                if line and len(line) > 3 and len(line) < 200:
                    facts.append(line)
            return facts[:8]
        except Exception as e:
            print(f"⚠️ 关键信息提取失败: {e}")
            return []

    def _merge_key_facts(
        self, existing: List[str], new_facts: List[str], max_facts: int = 10
    ) -> List[str]:
        """合并关键信息（去重 + 限制数量）"""
        all_facts = existing + new_facts

        # 简单去重（基于前20字符）
        seen = set()
        unique_facts = []
        for fact in all_facts:
            key = fact[:20].lower()
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)

        # 保留最新的 N 条（越新越重要）
        return unique_facts[-max_facts:]

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表格式化为文本"""
        parts = []
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            content = msg.get("content", "")[:500]  # 限制单条长度
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """估算消息列表的 token 数"""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // self.chars_per_token

    def reset(self):
        """重置压缩器状态"""
        self._current_summary = ""
        self._key_facts = []
        self._compression_count = 0

    @property
    def has_summary(self) -> bool:
        return bool(self._current_summary)

    @property
    def compression_count(self) -> int:
        return self._compression_count

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "has_summary": self.has_summary,
            "summary_length": len(self._current_summary),
            "key_facts_count": len(self._key_facts),
            "compression_count": self._compression_count,
        }
