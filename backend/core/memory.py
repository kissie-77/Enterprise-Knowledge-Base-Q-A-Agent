"""
记忆模块 - 管理对话上下文和会话记忆
参考 hello-agents 的 Memory 系统设计

支持三种记忆类型：
- 工作记忆（Working Memory）: 当前会话的短期记忆，随会话结束清除
- 情景记忆（Episodic Memory）: 用户交互历史，按时间顺序记录
- 语义记忆（Semantic Memory）: 长期知识和学习到的概念
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


class MemoryType(Enum):
    """记忆类型枚举"""
    WORKING = "working"       # 工作记忆（短期，当前会话）
    EPISODIC = "episodic"     # 情景记忆（中期，交互历史）
    SEMANTIC = "semantic"     # 语义记忆（长期，知识概念）


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5       # 重要性评分 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0         # 访问次数
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "session_id": self.session_id,
        }


class Memory:
    """
    对话记忆管理器

    功能：
    - 管理多轮对话上下文
    - 自动滑动窗口（限制 token 消耗）
    - 重要信息持久化
    - 记忆摘要生成

    使用方式:
        memory = Memory(max_turns=20)

        # 添加对话
        memory.add_user_message("什么是机器学习？")
        memory.add_assistant_message("机器学习是...")

        # 获取上下文（用于 LLM 调用）
        context_messages = memory.get_context_messages()

        # 添加重要知识
        memory.add_knowledge("用户关注深度学习方向", importance=0.8)
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_working_memory: int = 50,
        max_episodic_memory: int = 200,
        session_id: Optional[str] = None,
    ):
        """
        初始化记忆管理器

        Args:
            max_turns: 最大对话轮数（工作记忆中保留的轮数）
            max_working_memory: 工作记忆最大条目数
            max_episodic_memory: 情景记忆最大条目数
            session_id: 会话 ID
        """
        self.max_turns = max_turns
        self.session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")

        # 三种记忆存储
        self._working: deque = deque(maxlen=max_working_memory)
        self._episodic: deque = deque(maxlen=max_episodic_memory)
        self._semantic: List[MemoryEntry] = []

        # 对话历史（OpenAI 消息格式）
        self._conversation: deque = deque(maxlen=max_turns * 2)  # user + assistant = 2 条/轮

        # 系统提示
        self._system_prompt: Optional[str] = None

    # ===== 对话管理 =====

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self._system_prompt = prompt

    def add_user_message(self, content: str, importance: float = 0.5):
        """添加用户消息"""
        self._conversation.append({"role": "user", "content": content})

        entry = MemoryEntry(
            content=f"用户: {content}",
            memory_type=MemoryType.WORKING,
            importance=importance,
            session_id=self.session_id,
        )
        self._working.append(entry)

    def add_assistant_message(self, content: str, importance: float = 0.5):
        """添加助手回复"""
        self._conversation.append({"role": "assistant", "content": content})

        entry = MemoryEntry(
            content=f"助手: {content}",
            memory_type=MemoryType.WORKING,
            importance=importance,
            session_id=self.session_id,
        )
        self._working.append(entry)

    def add_tool_result(self, tool_name: str, result: str):
        """添加工具调用结果到记忆"""
        entry = MemoryEntry(
            content=f"工具[{tool_name}]: {result}",
            memory_type=MemoryType.WORKING,
            importance=0.6,
            metadata={"tool": tool_name},
            session_id=self.session_id,
        )
        self._working.append(entry)

    def get_context_messages(
        self,
        include_system: bool = True,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        获取用于 LLM 调用的上下文消息列表

        Args:
            include_system: 是否包含系统提示
            max_messages: 限制返回的最大消息数

        Returns:
            OpenAI 格式的消息列表
        """
        messages = []

        # 系统提示
        if include_system and self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # 对话历史
        conversation_list = list(self._conversation)
        if max_messages and len(conversation_list) > max_messages:
            conversation_list = conversation_list[-max_messages:]

        messages.extend(conversation_list)
        return messages

    def get_recent_context(self, num_turns: int = 5) -> str:
        """
        获取最近几轮对话的文本摘要

        Args:
            num_turns: 返回的最近轮数

        Returns:
            格式化的对话历史文本
        """
        conversation_list = list(self._conversation)
        recent = conversation_list[-(num_turns * 2):]

        parts = []
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else "助手"
            content_preview = msg["content"][:200]
            parts.append(f"{role_label}: {content_preview}")

        return "\n".join(parts) if parts else "（暂无对话历史）"

    # ===== 知识记忆 =====

    def add_knowledge(self, content: str, importance: float = 0.7, **metadata):
        """
        添加语义知识到长期记忆

        Args:
            content: 知识内容
            importance: 重要性（0-1）
            **metadata: 额外元数据
        """
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            metadata=metadata,
            session_id=self.session_id,
        )
        self._semantic.append(entry)

    def add_episode(self, content: str, importance: float = 0.6, **metadata):
        """
        添加情景记忆

        Args:
            content: 事件描述
            importance: 重要性
            **metadata: 额外元数据
        """
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            metadata=metadata,
            session_id=self.session_id,
        )
        self._episodic.append(entry)

    # ===== 记忆检索 =====

    def search_memory(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """
        搜索记忆（简单关键词匹配）

        Args:
            query: 搜索关键词
            memory_type: 限定记忆类型
            limit: 返回数量限制

        Returns:
            匹配的记忆条目列表
        """
        all_entries = []

        if memory_type is None or memory_type == MemoryType.WORKING:
            all_entries.extend(self._working)
        if memory_type is None or memory_type == MemoryType.EPISODIC:
            all_entries.extend(self._episodic)
        if memory_type is None or memory_type == MemoryType.SEMANTIC:
            all_entries.extend(self._semantic)

        # 简单关键词匹配 + 重要性排序
        matched = []
        query_lower = query.lower()
        for entry in all_entries:
            if query_lower in entry.content.lower():
                entry.access_count += 1
                matched.append(entry)

        # 按重要性和时间排序
        matched.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)
        return matched[:limit]

    def get_important_memories(self, threshold: float = 0.7, limit: int = 10) -> List[MemoryEntry]:
        """获取高重要性的记忆"""
        all_entries = list(self._working) + list(self._episodic) + self._semantic
        important = [e for e in all_entries if e.importance >= threshold]
        important.sort(key=lambda e: e.importance, reverse=True)
        return important[:limit]

    # ===== 会话管理 =====

    def clear_working_memory(self):
        """清空工作记忆（开始新话题时使用）"""
        self._working.clear()
        self._conversation.clear()

    def clear_all(self):
        """清空所有记忆"""
        self._working.clear()
        self._episodic.clear()
        self._semantic.clear()
        self._conversation.clear()

    def get_summary(self) -> Dict[str, Any]:
        """获取记忆状态摘要"""
        return {
            "session_id": self.session_id,
            "working_memory_count": len(self._working),
            "episodic_memory_count": len(self._episodic),
            "semantic_memory_count": len(self._semantic),
            "conversation_turns": len(self._conversation) // 2,
            "max_turns": self.max_turns,
        }

    @property
    def conversation_count(self) -> int:
        """当前对话轮数"""
        return len(self._conversation) // 2

    @property
    def is_empty(self) -> bool:
        """是否为空记忆"""
        return len(self._conversation) == 0

    def __repr__(self):
        return (
            f"<Memory(session='{self.session_id}', "
            f"turns={self.conversation_count}, "
            f"working={len(self._working)}, "
            f"episodic={len(self._episodic)}, "
            f"semantic={len(self._semantic)})>"
        )
