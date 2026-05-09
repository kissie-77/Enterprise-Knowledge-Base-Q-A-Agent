"""
Agent 基类模块
提供所有 Agent 的基础抽象
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Message:
    """对话消息"""
    content: str
    role: str  # "user", "assistant", "system", "tool"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseAgent(ABC):
    """
    Agent 基类
    所有智能体（SimpleAgent, ReActAgent 等）都继承此类
    """

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        max_history: int = 50,
    ):
        self.name = name
        self.system_prompt = system_prompt or "你是一个有用的 AI 助手。"
        self.max_history = max_history
        self._history: List[Message] = []
        self._created_at = datetime.now()

    @abstractmethod
    async def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Agent 处理用户输入

        Args:
            input_text: 用户输入
            **kwargs: 额外参数

        Returns:
            Agent 的回复
        """
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)
        # 保持历史记录在限制范围内
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_history(self) -> List[Message]:
        """获取对话历史"""
        return self._history.copy()

    def get_history_as_messages(self) -> List[Dict[str, str]]:
        """获取对话历史（OpenAI 消息格式）"""
        return [msg.to_dict() for msg in self._history]

    def clear_history(self):
        """清空对话历史"""
        self._history.clear()

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"
