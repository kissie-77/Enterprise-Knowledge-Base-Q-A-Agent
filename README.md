# 🤖 企业知识库问答 Agent

> 基于 [hello-agents](https://github.com/datawhalechina/hello-agents) 架构的企业级智能问答系统

本项目参考 Datawhale 社区 **hello-agents** 教程，从零构建了一个完整的 AI Native Agent 系统。集成了 **ReAct 推理范式** + **RAG 检索增强生成** + **多层记忆系统**，适用于企业知识库问答场景。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **ReAct 智能体** | Reasoning + Acting 多步推理，可解释的思考链 |
| 📚 **RAG 检索增强** | 多格式文档解析、智能分块、多查询扩展检索 |
| 🔧 **模块化工具系统** | 可插拔工具注册表，支持装饰器/继承/函数三种注册方式 |
| 💬 **多层记忆** | 工作记忆 + 情景记忆 + 语义记忆，支持多轮对话 |
| 🌐 **多 LLM 支持** | 智谱/DeepSeek/通义千问/Moonshot/OpenAI/硅基流动 |
| 📊 **评测面板** | 用户反馈收集、统计分析、趋势可视化 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit 前端                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ 问答界面  │  │ 评测面板  │  │  架构说明 / 配置  │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
└─────────────────────────┬───────────────────────────┘
                          │ HTTP API
┌─────────────────────────▼───────────────────────────┐
│                  FastAPI 后端                         │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │              ReAct Agent (核心)                  │  │
│  │                                                 │  │
│  │  Thought → Action → Observation → ... → Finish  │  │
│  └──────┬─────────────┬──────────────────┬────────┘  │
│         │             │                  │           │
│  ┌──────▼──────┐ ┌────▼─────┐  ┌────────▼────────┐  │
│  │ ToolRegistry │ │RAGEngine │  │     Memory      │  │
│  │             │ │          │  │                  │  │
│  │ • 时间工具  │ │ • 文档解析│  │ • 工作记忆      │  │
│  │ • 计算器   │ │ • 智能分块│  │ • 情景记忆      │  │
│  │ • 网页搜索  │ │ • 向量检索│  │ • 语义记忆      │  │
│  │ • 自定义... │ │ • MQE扩展 │  │ • 滑动窗口      │  │
│  └─────────────┘ └────┬─────┘  └─────────────────┘  │
│                       │                              │
│  ┌────────────────────▼─────────────────────────┐    │
│  │              LLMClient (通用)                  │    │
│  │  智谱 / DeepSeek / 通义 / Moonshot / OpenAI   │    │
│  └───────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────────┐  ┌────────────────────────────┐    │
│  │ ChromaDB     │  │ SQLite (反馈数据)           │    │
│  │ (向量存储)    │  │                            │    │
│  └──────────────┘  └────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
Enterprise-Knowledge-Base-Q-A-Agent/
├── backend/
│   ├── core/                    # 核心框架（hello-agents 架构）
│   │   ├── __init__.py          # 模块导出
│   │   ├── llm_client.py       # 通用 LLM 客户端
│   │   ├── tool_registry.py    # 工具注册与执行
│   │   ├── base_agent.py       # Agent 基类
│   │   ├── react_agent.py      # ReAct 智能体
│   │   ├── rag_engine.py       # RAG 检索增强引擎
│   │   ├── memory.py           # 记忆模块
│   │   └── tools/              # 内置工具集
│   │       ├── time_tool.py    # 时间工具
│   │       ├── calculator_tool.py  # 计算器
│   │       └── search_tool.py  # 网页搜索
│   ├── main.py                 # FastAPI 主入口
│   ├── feedback.py             # 反馈系统
│   ├── db.py                   # 数据库管理
│   └── .env.example            # 环境变量模板
├── frontend/
│   └── streamlit_app.py        # Streamlit 前端
├── requirements.txt            # Python 依赖
└── README.md                   # 项目文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cd backend
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

**最低配置**：只需要一个 LLM API Key 即可运行。推荐使用智谱 AI 的免费模型 `glm-4-flash`：

```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your_api_key_here
```

### 3. 启动后端

```bash
cd backend
python main.py
# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 启动前端

```bash
cd frontend
streamlit run streamlit_app.py
```

### 5. 开始使用

1. 打开浏览器访问 Streamlit 界面
2. 上传文档到知识库（支持 PDF/TXT/DOCX/MD）
3. 开始提问！

---

## 🔧 自定义扩展

### 添加自定义工具

```python
from core.tool_registry import BaseTool

class WeatherTool(BaseTool):
    """天气查询工具"""
    name = "weather"
    description = "查询指定城市的天气信息。输入城市名称即可。"

    async def execute(self, city: str = "", **kwargs) -> str:
        # 你的天气 API 调用逻辑
        return f"{city}今天晴，气温 25°C"

# 在 main.py 中注册
tool_registry.register(WeatherTool())
```

### 切换 LLM 提供商

只需修改 `.env` 文件：

```env
# 使用 DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key

# 使用通义千问
LLM_PROVIDER=qwen
QWEN_API_KEY=your_key

# 使用 OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

### 自定义 Agent 行为

```python
from core import ReActAgent, LLMClient, ToolRegistry

# 创建专业领域 Agent
agent = ReActAgent(
    name="法律助手",
    llm=LLMClient(provider="deepseek"),
    tool_registry=my_tools,
    max_steps=8,  # 允许更多推理步骤
)

# 运行
answer = await agent.run("合同违约的法律责任有哪些？", context=rag_context)
```

---

## 📖 参考资料

本项目基于以下资源构建：

- [hello-agents](https://github.com/datawhalechina/hello-agents) - Datawhale 智能体系统构建教程
  - 第4章：ReAct / Plan-and-Solve / Reflection 经典范式
  - 第7章：从零构建 Agent 框架
  - 第8章：Memory 与 RAG 检索增强
- [HelloAgents 框架](https://github.com/jjyaoao/HelloAgents) - hello-agents 配套 Agent 框架

---

## 📝 API 接口文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 |
| `/upload` | POST | 上传文档 |
| `/ask` | POST | 智能问答 |
| `/feedback` | POST | 提交反馈 |
| `/stats` | GET | 统计信息 |
| `/tools` | GET | 工具列表 |
| `/sessions/{id}` | GET | 会话信息 |
| `/sessions/{id}` | DELETE | 清除会话 |

### 问答接口详情

```json
POST /ask
{
    "question": "什么是机器学习？",
    "session_id": "user123",
    "use_rag": true,
    "use_agent": true,
    "top_k": 5
}

// 响应
{
    "status": "success",
    "answer": "机器学习是...",
    "contexts": [...],
    "tool_used": null,
    "reasoning_trace": "步骤1: 分析问题...",
    "response_time_ms": 1200,
    "session_id": "user123"
}
```

---

## 🤝 致谢

- [Datawhale](https://github.com/datawhalechina) 开源社区
- [hello-agents](https://github.com/datawhalechina/hello-agents) 教程团队
- 智谱 AI 提供免费 API 支持

---

## 📜 许可证

本项目基于 MIT 许可证开源。教程内容参考部分遵循 [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) 协议。
