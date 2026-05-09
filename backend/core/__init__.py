"""
企业知识库问答 Agent - 核心框架模块
基于 hello-agents 架构设计，实现模块化 Agent 系统

核心组件：
- LLMClient: 兼容 OpenAI 接口的 LLM 客户端
- ToolRegistry: 工具注册与执行器
- BaseAgent: Agent 基类
- ReActAgent: ReAct 范式智能体
- Memory: 对话记忆管理
- RAGEngine: 增强检索生成引擎

高级扩展模块（独创）：
- QueryRewriter: 查询改写（5种策略）
- HybridRetriever: 多路召回 + RRF 融合 + Reranker 精排
- CitationEngine: 回答可溯源 + 幻觉检测
- ReflectionEngine: Agent 自我反思与迭代改进
- ConversationCompressor: 对话摘要压缩
"""

from .llm_client import LLMClient
from .tool_registry import ToolRegistry, BaseTool
from .base_agent import BaseAgent, Message
from .react_agent import ReActAgent
from .memory import Memory, MemoryType
from .rag_engine import RAGEngine
from .query_rewriter import QueryRewriter, RewriteResult
from .retriever import HybridRetriever, Reranker, BM25Retriever, rrf_fusion
from .citation import CitationEngine, CitedAnswer
from .reflection import ReflectionEngine, ReflectionResult
from .conversation_compressor import ConversationCompressor, CompressedContext

__all__ = [
    # 基础组件
    "LLMClient",
    "ToolRegistry",
    "BaseTool",
    "BaseAgent",
    "Message",
    "ReActAgent",
    "Memory",
    "MemoryType",
    "RAGEngine",
    # 高级扩展
    "QueryRewriter",
    "RewriteResult",
    "HybridRetriever",
    "Reranker",
    "BM25Retriever",
    "rrf_fusion",
    "CitationEngine",
    "CitedAnswer",
    "ReflectionEngine",
    "ReflectionResult",
    "ConversationCompressor",
    "CompressedContext",
]
