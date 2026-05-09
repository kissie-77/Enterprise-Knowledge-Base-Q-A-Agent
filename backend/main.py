"""
企业知识库问答 Agent - FastAPI 后端主入口
基于 hello-agents 架构重构，集成 ReAct 智能体 + RAG + Memory
"""
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from core import LLMClient, ToolRegistry, ReActAgent, RAGEngine, Memory
from core.tools import TimeTool, CalculatorTool, WebSearchTool
from feedback import save_feedback, get_feedback_stats, init_feedback_db

# ===== 全局组件 =====
llm: Optional[LLMClient] = None
rag_engine: Optional[RAGEngine] = None
tool_registry: Optional[ToolRegistry] = None
agent: Optional[ReActAgent] = None

# 会话记忆存储（session_id -> Memory）
sessions: Dict[str, Memory] = {}

# 限流器
limiter = Limiter(key_func=get_remote_address)


def init_agent_system():
    """初始化智能体系统"""
    global llm, rag_engine, tool_registry, agent

    # 1. 初始化 LLM 客户端
    provider = os.getenv("LLM_PROVIDER", "zhipu")
    try:
        llm = LLMClient(provider=provider)
    except ValueError as e:
        print(f"⚠️ LLM 初始化失败 (provider={provider}): {e}")
        print("  请在 .env 中配置 LLM_API_KEY / ZHIPU_API_KEY 等环境变量")
        llm = None

    # 2. 初始化 RAG 引擎
    rag_engine = RAGEngine(
        chroma_path=os.getenv("CHROMA_PATH", "chroma_store"),
        collection_name=os.getenv("COLLECTION_NAME", "knowledge_base"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )

    # 3. 注册工具
    tool_registry = ToolRegistry()
    tool_registry.register(TimeTool())
    tool_registry.register(CalculatorTool())
    tool_registry.register(WebSearchTool())

    # 4. 创建 ReAct 智能体
    if llm:
        agent = ReActAgent(
            name="企业知识库问答Agent",
            llm=llm,
            tool_registry=tool_registry,
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "5")),
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        )

    print("✅ 智能体系统初始化完成")


def get_or_create_session(session_id: str) -> Memory:
    """获取或创建会话记忆"""
    if session_id not in sessions:
        sessions[session_id] = Memory(
            max_turns=20,
            session_id=session_id,
        )
        sessions[session_id].set_system_prompt(
            "你是一个专业的企业知识库问答助手。你可以根据知识库中的文档内容回答问题，"
            "也可以使用工具来获取实时信息。请详细、准确地回答用户的问题。"
        )
    return sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    init_feedback_db()
    os.makedirs("data", exist_ok=True)
    os.makedirs("chroma_store", exist_ok=True)
    init_agent_system()
    print("🚀 服务已启动")
    yield
    # 关闭
    print("👋 服务关闭")


app = FastAPI(
    title="企业知识库问答 Agent",
    description="基于 ReAct + RAG 的智能问答系统（hello-agents 架构）",
    version="2.0.0",
    lifespan=lifespan,
)

# 限流中间件
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== API 路由 =====


@app.get("/")
async def root():
    """服务信息"""
    return {
        "service": "企业知识库问答 Agent",
        "version": "2.0.0",
        "architecture": "ReAct + RAG + Memory (hello-agents)",
        "status": "running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "llm_ready": llm is not None,
        "tools": tool_registry.list_tools() if tool_registry else [],
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "llm": "connected" if llm else "not configured",
        "rag": "ready" if rag_engine else "not ready",
        "agent": "ready" if agent else "not ready",
    }


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    overlap: int = Form(50),
):
    """
    上传文档并向量化存储

    支持格式: PDF, TXT, DOCX, MD
    """
    try:
        # 验证文件类型
        allowed_extensions = {".pdf", ".txt", ".docx", ".md"}
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}。支持: {', '.join(allowed_extensions)}",
            )

        # 保存文件
        file_path = f"data/{file.filename}"
        os.makedirs("data", exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 使用 RAG 引擎处理文档
        chunk_count = await rag_engine.add_document(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            metadata={"filename": file.filename, "upload_time": datetime.now().isoformat()},
        )

        return {
            "status": "success",
            "filename": file.filename,
            "chunk_count": chunk_count,
            "message": f"文档已成功处理，切分为 {chunk_count} 个片段",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@app.post("/ask")
@limiter.limit("30/minute")
async def ask_question(request: Request, question_data: dict):
    """
    智能问答接口（ReAct Agent + RAG）

    请求体:
        {
            "question": "用户问题",
            "session_id": "会话ID（可选，用于多轮对话）",
            "use_rag": true,        // 是否使用知识库检索（默认 true）
            "use_agent": true,      // 是否使用 ReAct 推理（默认 true）
            "top_k": 5              // 检索结果数量
        }
    """
    try:
        question = question_data.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        session_id = question_data.get("session_id", "default")
        use_rag = question_data.get("use_rag", True)
        use_agent = question_data.get("use_agent", True)
        top_k = question_data.get("top_k", 5)

        if not agent and not llm:
            raise HTTPException(status_code=503, detail="LLM 服务未配置，请检查 API Key")

        start_time = time.time()

        # 获取会话记忆
        memory = get_or_create_session(session_id)
        memory.add_user_message(question)

        # RAG 检索
        context = ""
        contexts = []
        if use_rag and rag_engine:
            results = await rag_engine.search(question, top_k=top_k)
            if results:
                context = rag_engine.build_context(results)
                contexts = [{"content": r["content"], "score": r["score"]} for r in results]

        # 生成回答
        tool_used = None
        reasoning_trace = None

        if use_agent and agent:
            # 使用 ReAct 智能体（推理 + 工具调用 + RAG）
            answer = await agent.run(input_text=question, context=context)
            reasoning_trace = agent.get_trace_summary()

            # 检查是否使用了工具
            trace = agent.get_reasoning_trace()
            for step in trace:
                action = step.get("action", "")
                if action and not action.startswith("Finish"):
                    tool_used = action.split("[")[0] if "[" in action else action
                    break
        else:
            # 降级模式：直接 LLM 调用
            if context:
                prompt = f"""基于以下参考资料回答用户问题。

参考资料:
{context}

用户问题: {question}

请详细、准确地回答："""
            else:
                prompt = question

            answer = await llm.think(prompt)

        # 保存到记忆
        memory.add_assistant_message(answer)

        response_time = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "tool_used": tool_used,
            "reasoning_trace": reasoning_trace,
            "response_time_ms": response_time,
            "retrieved_chunks": len(contexts),
            "session_id": session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@app.post("/feedback")
@limiter.limit("60/minute")
async def submit_feedback(request: Request, feedback_data: dict):
    """提交用户反馈"""
    try:
        question = feedback_data.get("question", "")
        answer = feedback_data.get("answer", "")
        feedback = feedback_data.get("feedback", "")

        if not question or not answer or not feedback:
            raise HTTPException(status_code=400, detail="缺少必要的反馈信息")

        if feedback not in ["upvote", "downvote"]:
            raise HTTPException(status_code=400, detail="反馈类型必须是 'upvote' 或 'downvote'")

        save_feedback(
            question=question,
            answer=answer,
            feedback=feedback,
            response_time_ms=feedback_data.get("response_time_ms", 0),
            retrieved_chunks=feedback_data.get("retrieved_chunks", 0),
        )

        return {"status": "success", "message": "反馈已记录"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反馈保存失败: {str(e)}")


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        feedback_stats = get_feedback_stats()
        rag_stats = rag_engine.get_stats() if rag_engine else {}

        return {
            "status": "success",
            "stats": feedback_stats,
            "rag": rag_stats,
            "active_sessions": len(sessions),
            "tools": tool_registry.list_tools() if tool_registry else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    memory = sessions[session_id]
    return {
        "status": "success",
        "session": memory.get_summary(),
        "recent_context": memory.get_recent_context(num_turns=5),
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """清除会话"""
    if session_id in sessions:
        sessions[session_id].clear_all()
        del sessions[session_id]
    return {"status": "success", "message": f"会话 {session_id} 已清除"}


@app.get("/tools")
async def list_tools():
    """列出所有可用工具"""
    if not tool_registry:
        return {"tools": []}

    tools_info = []
    for name in tool_registry.list_tools():
        tool = tool_registry.get_tool(name)
        tools_info.append({
            "name": tool.name,
            "description": tool.description,
        })

    return {"status": "success", "tools": tools_info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
