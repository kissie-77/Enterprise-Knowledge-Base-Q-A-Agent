"""
企业知识库问答 Agent - FastAPI 后端主入口
基于 hello-agents 架构 + 6大高级扩展模块

完整处理管道:
用户提问 → 查询改写 → 多路召回+重排序 → ReAct推理+工具调用
         → 生成带引用回答 → 自我反思改进 → 对话摘要压缩 → 返回
"""
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from core import (
    LLMClient, ToolRegistry, ReActAgent, RAGEngine, Memory,
    QueryRewriter, HybridRetriever, Reranker,
    CitationEngine, ReflectionEngine, ConversationCompressor,
)
from core.tools import TimeTool, CalculatorTool, WebSearchTool
from feedback import save_feedback, get_feedback_stats, init_feedback_db

# ===== 全局组件 =====
llm: Optional[LLMClient] = None
rag_engine: Optional[RAGEngine] = None
tool_registry: Optional[ToolRegistry] = None
agent: Optional[ReActAgent] = None

# 高级模块
query_rewriter: Optional[QueryRewriter] = None
hybrid_retriever: Optional[HybridRetriever] = None
citation_engine: Optional[CitationEngine] = None
reflection_engine: Optional[ReflectionEngine] = None

# 会话管理（session_id -> {memory, compressor}）
sessions: Dict[str, Dict] = {}

# 限流器
limiter = Limiter(key_func=get_remote_address)


def init_agent_system():
    """初始化完整智能体系统"""
    global llm, rag_engine, tool_registry, agent
    global query_rewriter, hybrid_retriever, citation_engine, reflection_engine

    # 1. 初始化 LLM 客户端
    provider = os.getenv("LLM_PROVIDER", "zhipu")
    try:
        llm = LLMClient(provider=provider)
    except ValueError as e:
        print(f"⚠️ LLM 初始化失败 (provider={provider}): {e}")
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

    # 4. 初始化高级模块（需要 LLM）
    if llm:
        # 查询改写器
        query_rewriter = QueryRewriter(
            llm=llm,
            default_strategy="auto",
            num_sub_queries=3,
            enable_hyde=os.getenv("ENABLE_HYDE", "true").lower() == "true",
        )

        # 混合检索器（多路召回 + Reranker）
        reranker = Reranker(
            use_llm_fallback=True,
            llm=llm,
        )
        hybrid_retriever = HybridRetriever(
            rag_engine=rag_engine,
            query_rewriter=query_rewriter,
            reranker=reranker,
            enable_bm25=True,
            enable_reranker=os.getenv("ENABLE_RERANKER", "true").lower() == "true",
        )

        # 引用溯源引擎
        citation_engine = CitationEngine(llm=llm)

        # 反思改进引擎
        reflection_engine = ReflectionEngine(
            llm=llm,
            max_iterations=int(os.getenv("REFLECTION_MAX_ITER", "2")),
            quality_threshold=float(os.getenv("REFLECTION_THRESHOLD", "7.0")),
        )

        # 创建 ReAct 智能体
        agent = ReActAgent(
            name="企业知识库问答Agent",
            llm=llm,
            tool_registry=tool_registry,
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "5")),
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        )

    print("✅ 智能体系统初始化完成（含6大高级模块）")


def get_or_create_session(session_id: str) -> Dict:
    """获取或创建会话（含 Memory + Compressor）"""
    if session_id not in sessions:
        memory = Memory(max_turns=20, session_id=session_id)
        memory.set_system_prompt(
            "你是一个专业的企业知识库问答助手。你可以根据知识库中的文档内容回答问题，"
            "也可以使用工具来获取实时信息。请详细、准确地回答用户的问题。"
        )
        compressor = ConversationCompressor(
            llm=llm,
            window_size=int(os.getenv("COMPRESSOR_WINDOW", "6")),
            max_tokens_estimate=int(os.getenv("COMPRESSOR_MAX_TOKENS", "4000")),
        ) if llm else None

        sessions[session_id] = {
            "memory": memory,
            "compressor": compressor,
        }
    return sessions[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    init_feedback_db()
    os.makedirs("data", exist_ok=True)
    os.makedirs("chroma_store", exist_ok=True)
    init_agent_system()
    print("🚀 服务已启动")
    yield
    print("👋 服务关闭")


app = FastAPI(
    title="企业知识库问答 Agent",
    description="基于 hello-agents + 6大高级扩展的智能问答系统",
    version="3.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
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
    return {
        "service": "企业知识库问答 Agent",
        "version": "3.0.0",
        "architecture": "ReAct + RAG + 6大高级扩展 (hello-agents)",
        "modules": {
            "query_rewriter": query_rewriter is not None,
            "hybrid_retriever": hybrid_retriever is not None,
            "citation_engine": citation_engine is not None,
            "reflection_engine": reflection_engine is not None,
            "conversation_compressor": True,
        },
        "status": "running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm": "connected" if llm else "not configured",
        "rag": "ready" if rag_engine else "not ready",
        "agent": "ready" if agent else "not ready",
        "advanced_modules": {
            "query_rewriter": "ready" if query_rewriter else "disabled",
            "hybrid_retriever": "ready" if hybrid_retriever else "disabled",
            "citation_engine": "ready" if citation_engine else "disabled",
            "reflection_engine": "ready" if reflection_engine else "disabled",
        },
    }


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    overlap: int = Form(50),
):
    """上传文档并向量化存储"""
    try:
        allowed_extensions = {".pdf", ".txt", ".docx", ".md"}
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")

        file_path = f"data/{file.filename}"
        os.makedirs("data", exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        chunk_count = await rag_engine.add_document(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            metadata={"filename": file.filename, "upload_time": datetime.now().isoformat()},
        )

        # 使 BM25 缓存失效（新文档需要重建索引）
        if hybrid_retriever:
            hybrid_retriever.invalidate_bm25_cache()

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
    智能问答接口 - 完整管道

    请求体:
    {
        "question": "用户问题",
        "session_id": "会话ID",
        "use_rag": true,
        "use_agent": true,
        "use_rewrite": true,         // 是否启用查询改写
        "use_hybrid_retrieval": true, // 是否启用混合检索
        "use_citation": true,         // 是否启用引用溯源
        "use_reflection": false,      // 是否启用反思改进（较慢）
        "rewrite_strategy": "auto",   // 改写策略
        "top_k": 5
    }
    """
    try:
        question = question_data.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        if not agent and not llm:
            raise HTTPException(status_code=503, detail="LLM 服务未配置")

        session_id = question_data.get("session_id", "default")
        use_rag = question_data.get("use_rag", True)
        use_agent = question_data.get("use_agent", True)
        use_rewrite = question_data.get("use_rewrite", True)
        use_hybrid = question_data.get("use_hybrid_retrieval", True)
        use_citation = question_data.get("use_citation", True)
        use_reflection = question_data.get("use_reflection", False)
        rewrite_strategy = question_data.get("rewrite_strategy", "auto")
        top_k = question_data.get("top_k", 5)

        start_time = time.time()
        session = get_or_create_session(session_id)
        memory = session["memory"]
        compressor = session["compressor"]

        # 记录用户消息
        memory.add_user_message(question)

        # 管道结果收集
        pipeline_info = {}

        # ===== 步骤1: 对话压缩（如有需要）=====
        conversation_history = memory.get_recent_context(num_turns=5)

        # ===== 步骤2: 查询改写 =====
        rewrite_info = None
        effective_query = question

        if use_rewrite and query_rewriter:
            rewrite_result = await query_rewriter.rewrite(
                query=question,
                strategy=rewrite_strategy,
                conversation_history=conversation_history,
            )
            effective_query = rewrite_result.rewritten_query
            rewrite_info = {
                "original": question,
                "rewritten": effective_query,
                "strategy": rewrite_result.strategy_used,
                "sub_queries": rewrite_result.sub_queries,
                "context_used": rewrite_result.context_used,
            }
            pipeline_info["query_rewrite"] = rewrite_info

        # ===== 步骤3: 检索 =====
        context = ""
        contexts = []
        retrieval_info = None

        if use_rag and rag_engine:
            if use_hybrid and hybrid_retriever:
                # 高级：混合检索管道
                retrieval_result = await hybrid_retriever.retrieve(
                    query=effective_query,
                    top_k=top_k,
                    rewrite_strategy="simple" if use_rewrite else "auto",
                    conversation_history=conversation_history,
                )
                context = retrieval_result.get_context()
                contexts = [r.to_dict() for r in retrieval_result.final_results]
                retrieval_info = {
                    "methods": retrieval_result.methods_used,
                    "initial_candidates": retrieval_result.initial_candidates,
                    "reranker_used": retrieval_result.reranker_used,
                    "final_count": len(retrieval_result.final_results),
                }
            else:
                # 基础：纯向量检索
                results = await rag_engine.search(effective_query, top_k=top_k)
                if results:
                    context = rag_engine.build_context(results)
                    contexts = [{"content": r["content"], "score": r["score"], "metadata": r.get("metadata", {})} for r in results]

            pipeline_info["retrieval"] = retrieval_info

        # ===== 步骤4: 生成回答 =====
        tool_used = None
        reasoning_trace = None

        if use_agent and agent:
            answer = await agent.run(input_text=question, context=context)
            reasoning_trace = agent.get_trace_summary()
            trace = agent.get_reasoning_trace()
            for step in trace:
                action = step.get("action", "")
                if action and not action.startswith("Finish"):
                    tool_used = action.split("[")[0] if "[" in action else action
                    break
        elif llm:
            if context:
                prompt = f"基于以下参考资料回答用户问题。\n\n参考资料:\n{context}\n\n用户问题: {question}\n\n请详细、准确地回答："
            else:
                prompt = question
            answer = await llm.think(prompt)
        else:
            answer = "LLM 服务不可用"

        # ===== 步骤5: 引用溯源 =====
        citation_info = None
        if use_citation and citation_engine and contexts:
            context_for_citation = [
                {"content": c.get("content", ""), "metadata": c.get("metadata", {})}
                for c in contexts
            ]
            cited = await citation_engine.generate_with_citations(
                question=question,
                contexts=context_for_citation,
            )
            # 用带引用的回答替换原回答
            if cited.formatted_answer and cited.citations:
                answer = cited.formatted_answer
                citation_info = {
                    "citation_count": len(cited.citations),
                    "sources": cited.sources,
                    "coverage_score": cited.coverage_score,
                    "hallucination_risk": cited.hallucination_risk,
                }
                pipeline_info["citation"] = citation_info

        # ===== 步骤6: 自我反思改进 =====
        reflection_info = None
        if use_reflection and reflection_engine:
            ref_result = await reflection_engine.reflect_and_refine(
                question=question,
                initial_answer=answer,
                context=context,
            )
            if ref_result.was_refined:
                answer = ref_result.final_answer
            reflection_info = {
                "was_refined": ref_result.was_refined,
                "iterations": ref_result.iterations,
                "final_score": ref_result.final_score,
                "converged": ref_result.converged,
            }
            pipeline_info["reflection"] = reflection_info

        # ===== 步骤7: 保存到记忆 + 压缩 =====
        memory.add_assistant_message(answer)

        compression_info = None
        if compressor:
            messages = [m for m in memory.get_context_messages(include_system=False)]
            compressed = await compressor.compress_if_needed(messages)
            if compressed.compression_ratio < 1.0:
                compression_info = compressed.to_dict()
                pipeline_info["compression"] = compression_info

        # ===== 返回结果 =====
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
            # 高级管道信息
            "pipeline": pipeline_info,
            "query_rewrite": rewrite_info,
            "citation": citation_info,
            "reflection": reflection_info,
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
            raise HTTPException(status_code=400, detail="反馈类型必须是 upvote 或 downvote")

        save_feedback(
            question=question, answer=answer, feedback=feedback,
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
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    session = sessions[session_id]
    memory = session["memory"]
    compressor = session["compressor"]
    return {
        "status": "success",
        "session": memory.get_summary(),
        "recent_context": memory.get_recent_context(num_turns=5),
        "compressor_state": compressor.get_state() if compressor else None,
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """清除会话"""
    if session_id in sessions:
        sessions[session_id]["memory"].clear_all()
        if sessions[session_id]["compressor"]:
            sessions[session_id]["compressor"].reset()
        del sessions[session_id]
    return {"status": "success", "message": f"会话 {session_id} 已清除"}


@app.get("/tools")
async def list_tools():
    """列出所有可用工具"""
    if not tool_registry:
        return {"tools": []}
    return {
        "status": "success",
        "tools": [
            {"name": tool_registry.get_tool(n).name, "description": tool_registry.get_tool(n).description}
            for n in tool_registry.list_tools()
        ],
    }


@app.get("/pipeline/config")
async def get_pipeline_config():
    """获取当前管道配置"""
    return {
        "status": "success",
        "config": {
            "llm_provider": os.getenv("LLM_PROVIDER", "zhipu"),
            "llm_model": llm.model if llm else None,
            "query_rewriter": {
                "enabled": query_rewriter is not None,
                "default_strategy": query_rewriter.default_strategy if query_rewriter else None,
                "hyde_enabled": query_rewriter.enable_hyde if query_rewriter else False,
            },
            "hybrid_retriever": {
                "enabled": hybrid_retriever is not None,
                "bm25_enabled": hybrid_retriever.enable_bm25 if hybrid_retriever else False,
                "reranker_enabled": hybrid_retriever.enable_reranker if hybrid_retriever else False,
            },
            "citation_engine": {"enabled": citation_engine is not None},
            "reflection_engine": {
                "enabled": reflection_engine is not None,
                "max_iterations": reflection_engine.max_iterations if reflection_engine else 0,
                "quality_threshold": reflection_engine.quality_threshold if reflection_engine else 0,
            },
            "conversation_compressor": {
                "enabled": True,
                "window_size": int(os.getenv("COMPRESSOR_WINDOW", "6")),
            },
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
