# 🤖 企业知识库问答 Agent（高级版）

> 基于 [hello-agents](https://github.com/datawhalechina/hello-agents) 架构 + **6大独创高级模块**的企业级智能问答系统

本项目在 Datawhale 社区 hello-agents 教程基础上，深度扩展了 **查询改写、多路召回、引用溯源、自我反思、对话压缩** 等工程能力，展示从"教程 Demo"到"生产级系统"的完整进阶路径。

---

## 🎯 项目亮点

| 扩展模块 | 解决的核心问题 | 相关概念 |
|---------|---------------|---------|
| **查询改写** | 用户提问模糊，检索召回率低 | HyDE 原理、Multi-Query 策略 |
| **多路召回 + Reranker** | 单一向量检索精度不足 | Bi-Encoder vs Cross-Encoder |
| **引用溯源** | RAG 回答不可信、无法审计 | 幻觉检测、Attribution 方法 |
| **Agent 反思** | 一次生成质量不稳定 | Self-Refine 论文、收敛条件设计 |
| **对话压缩** | 长对话 token 爆炸 | 增量摘要 vs 截断的信息保留率 |
| **ReAct 推理** | 复杂问题需要多步推理 | Reasoning + Acting 范式 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit 前端                            │
│  问答界面 │ 管道可视化 │ 评测面板 │ 架构说明                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP API
┌────────────────────────────▼────────────────────────────────────┐
│                     完整问答管道 (Pipeline)                       │
│                                                                  │
│  ① 查询改写 ──→ ② 多路召回 ──→ ③ ReAct推理 ──→ ④ 引用生成      │
│  (QueryRewriter) (HybridRetriever) (ReActAgent)  (CitationEngine)│
│       │                │                              │          │
│       ▼                ▼                              ▼          │
│  ⑤ 自我反思 ←── 回答生成 ←────────────────── ⑥ 对话压缩         │
│  (Reflection)                              (Compressor)          │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                       基础设施层                                  │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────────────┐  │
│  │LLMClient │  │ RAGEngine│  │ Memory │  │  ToolRegistry    │  │
│  │多厂商兼容 │  │ChromaDB  │  │三层记忆 │  │ 时间/计算/搜索  │  │
│  └──────────┘  └──────────┘  └────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
backend/
├── core/                          # 核心框架
│   ├── __init__.py                # 统一导出
│   ├── llm_client.py             # 通用 LLM 客户端（6大厂商）
│   ├── tool_registry.py          # 可插拔工具注册系统
│   ├── base_agent.py             # Agent 基类
│   ├── react_agent.py            # ReAct 推理智能体
│   ├── rag_engine.py             # RAG 向量检索引擎
│   ├── memory.py                 # 三层记忆系统
│   │
│   │── # ===== 以下为独创高级模块 =====
│   ├── query_rewriter.py         # ★ 查询改写（5种策略）
│   ├── retriever.py              # ★ 多路召回 + RRF + Reranker
│   ├── citation.py               # ★ 引用溯源 + 幻觉检测
│   ├── reflection.py             # ★ 自我反思 + 迭代改进
│   ├── conversation_compressor.py # ★ 对话摘要压缩
│   │
│   └── tools/                    # 内置工具集
│       ├── time_tool.py
│       ├── calculator_tool.py
│       └── search_tool.py
├── main.py                       # FastAPI 主入口（完整管道编排）
├── feedback.py                   # 反馈系统
├── db.py                         # 数据库
└── .env.example                  # 环境变量模板
frontend/
└── streamlit_app.py              # 前端界面
```

---

## 🔬 各模块技术深度

### 1. 查询改写 (Query Rewriting)

**问题**：用户输入"这个怎么用"→ 向量检索不知道"这个"是什么

**方案**：5种策略自动选择
- `simple`: 去口语化 + 关键词优化
- `multi_query`: 一个问题拆解为3个角度的子查询
- `hyde`: 先生成假设性答案文档，再用其向量去检索（HyDE 方法）
- `context_aware`: 结合多轮对话历史，补全指代和省略
- `full`: 完整流水线组合

```python
rewriter = QueryRewriter(llm=llm, default_strategy="auto")
result = await rewriter.rewrite("它的优缺点是什么？", conversation_history="...")
# result.rewritten_query = "RAG检索增强生成技术的优缺点分析"
```

### 2. 多路召回 + Reranker

**问题**：纯向量检索对关键词、数字、专有名词匹配差

**方案**：
- **语义路**（向量）：捕捉语义相似
- **关键词路**（BM25）：精确匹配专有名词
- **RRF 融合**：不同检索分数量纲不同，用倒数排名融合
- **Cross-Encoder 精排**：候选集精细化打分（比 Bi-Encoder 更准确）

```
Query → [向量检索 × N查询] + [BM25检索] → RRF融合 → Reranker精排 → Top-K
```

### 3. 引用溯源 (Citation)

**问题**：RAG 回答"像模像样"但可能是幻觉，用户无法验证

**方案**：
- **Inline Citation**：生成时就要求标注 `[1]` `[2]`
- **Post-hoc Attribution**：事后对每句话归因到来源片段
- **幻觉检测**：LLM 自动评估回答与参考资料的一致性

```python
cited = await citation_engine.generate_with_citations(question, contexts)
# cited.formatted_answer = "RAG是一种检索增强技术[1]，它结合了...[2]"
# cited.hallucination_risk = 0.1  # 低幻觉风险
```

### 4. 自我反思 (Reflection)

**问题**：LLM 一次生成的质量不稳定

**方案**（参考 Self-Refine 论文）：
- 生成 → 评审（Critic）→ 改进 → 再评审 → 收敛或达到上限
- 4维评分：准确性、完整性、清晰度、相关性
- 收敛条件：分数 ≥ 阈值 或 改进幅度 < 最小改进值

```python
result = await reflection_engine.reflect_and_refine(question, answer, context)
# result.iterations = 2, result.final_score = 8.5
```

### 5. 对话压缩 (Conversation Compression)

**问题**：20轮对话后 token 爆炸，简单截断丢失早期重要信息

**方案**：
- 滑动窗口：最近 N 轮保持原文
- 增量摘要：旧对话压缩为摘要（不重新处理全部历史）
- 关键信息提取：实体、结论、偏好单独保留

```
原始 20 轮对话 (8000 tokens) → 压缩后 (2000 tokens)
  = 摘要(200字) + 关键信息(8条) + 最近6轮原文
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cd backend && cp .env.example .env
# 编辑 .env，至少填入一个 LLM API Key

# 3. 启动后端
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. 启动前端
cd ../frontend && streamlit run streamlit_app.py
```

**最低配置**：智谱 AI 免费模型 `glm-4-flash`
```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your_key_here
```

---

## 📡 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务信息 + 模块状态 |
| `/health` | GET | 健康检查 |
| `/upload` | POST | 上传文档 |
| `/ask` | POST | **智能问答（完整管道）** |
| `/feedback` | POST | 用户反馈 |
| `/stats` | GET | 统计信息 |
| `/pipeline/config` | GET | 管道配置详情 |
| `/sessions/{id}` | GET/DELETE | 会话管理 |
| `/tools` | GET | 工具列表 |

### /ask 接口详情

```json
POST /ask
{
    "question": "什么是向量数据库？",
    "session_id": "user_123",
    "use_rag": true,
    "use_agent": true,
    "use_rewrite": true,
    "use_hybrid_retrieval": true,
    "use_citation": true,
    "use_reflection": false,
    "rewrite_strategy": "auto",
    "top_k": 5
}
```

**响应**（包含完整管道信息）：
```json
{
    "status": "success",
    "answer": "向量数据库是...[1]...[2]",
    "contexts": [...],
    "pipeline": {
        "query_rewrite": {"original": "...", "rewritten": "...", "strategy": "simple"},
        "retrieval": {"methods": ["vector", "bm25"], "reranker_used": true},
        "citation": {"citation_count": 3, "coverage_score": 0.85},
        "reflection": {"was_refined": true, "final_score": 8.2}
    },
    "response_time_ms": 2300
}
```

---

## 🎓 面试讲解思路

### "介绍一下你的项目"

> 这是一个基于 hello-agents 教程的企业知识库 RAG 问答系统。
> 在教程基础架构（ReAct + RAG）上，我独立实现了6个高级模块，
> 解决了 RAG 系统从"能用"到"好用"的关键工程问题。

### "你遇到了什么技术难点？"

1. **查询改写**：如何让"自动策略选择"在大多数场景下做出正确判断
2. **RRF 融合**：不同检索路的分数量纲不同，不能简单相加
3. **反思收敛**：如何防止 Reflection 死循环（设计了分数阈值 + 最小改进幅度双重条件）
4. **对话压缩**：增量摘要 vs 全量重新摘要的 trade-off

### "为什么选择这些模块？"

> 这5个扩展对应 RAG 系统最常见的5个痛点：
> 检索不准（改写+多路）、回答不可信（引用）、质量不稳定（反思）、长对话炸token（压缩）。
> 每个都是实际部署时绕不开的问题。

---

## 📖 参考资料

- [hello-agents](https://github.com/datawhalechina/hello-agents) - Datawhale Agent 教程（第4/7/8/12章）
- [Self-Refine](https://arxiv.org/abs/2303.17651) - Iterative Refinement with Self-Feedback (Madaan et al., 2023)
- [HyDE](https://arxiv.org/abs/2212.10496) - Hypothetical Document Embeddings (Gao et al., 2022)
- [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Reciprocal Rank Fusion

---

## 📜 许可证

MIT License
