"""
企业知识库问答 Agent - FastAPI 后端主入口
"""
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from upload import process_document
from ask import answer_question
from feedback import save_feedback, get_feedback_stats
from db import init_db

# 限流器配置
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    init_db()
    print("✅ 数据库初始化完成")
    yield
    # 关闭时清理
    print("👋 服务关闭")


app = FastAPI(
    title="企业知识库问答 Agent",
    description="基于 RAG 的智能问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加限流中间件
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "企业知识库问答 Agent",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    overlap: int = Form(50)
):
    """
    上传文档并进行向量化处理

    Args:
        file: 上传的文件（PDF/TXT/DOCX）
        chunk_size: 文本切片大小（默认500字符）
        overlap: 切片重叠字符数（默认50字符）
    """
    try:
        # 验证文件类型
        allowed_types = [
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="只支持 PDF、TXT 和 DOCX 文件"
            )

        # 保存文件到临时目录
        file_path = f"data/{file.filename}"
        os.makedirs("data", exist_ok=True)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 处理文档并存入向量库
        chunk_count = await process_document(file_path, chunk_size, overlap)

        return {
            "status": "success",
            "filename": file.filename,
            "chunk_count": chunk_count,
            "message": f"文档已成功处理，切分为 {chunk_count} 个片段"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败: {str(e)}"
        )


@app.post("/ask")
@limiter.limit("30/minute")
async def ask_question(request: Request, question_data: dict):
    """
    RAG 问答接口

    Args:
        question_data: {"question": "用户问题"}
    """
    try:
        question = question_data.get("question", "").strip()
        if not question:
            raise HTTPException(
                status_code=400,
                detail="问题不能为空"
            )

        start_time = time.time()
        result = await answer_question(question)
        response_time = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "question": question,
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "tool_used": result.get("tool_used", None),
            "response_time_ms": response_time,
            "retrieved_chunks": len(result.get("contexts", []))
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"问答处理失败: {str(e)}"
        )


@app.post("/feedback")
@limiter.limit("60/minute")
async def submit_feedback(request: Request, feedback_data: dict):
    """
    提交用户反馈

    Args:
        feedback_data: {
            "question_id": "问题ID（可选）",
            "question": "用户问题",
            "answer": "模型回答",
            "feedback": "upvote/downvote",
            "response_time_ms": 响应时间,
            "retrieved_chunks": 检索到的片段数
        }
    """
    try:
        question = feedback_data.get("question", "")
        answer = feedback_data.get("answer", "")
        feedback = feedback_data.get("feedback", "")

        if not question or not answer or not feedback:
            raise HTTPException(
                status_code=400,
                detail="缺少必要的反馈信息"
            )

        if feedback not in ["upvote", "downvote"]:
            raise HTTPException(
                status_code=400,
                detail="反馈类型必须是 'upvote' 或 'downvote'"
            )

        save_feedback(
            question=question,
            answer=answer,
            feedback=feedback,
            response_time_ms=feedback_data.get("response_time_ms", 0),
            retrieved_chunks=feedback_data.get("retrieved_chunks", 0)
        )

        return {
            "status": "success",
            "message": "反馈已记录"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"反馈保存失败: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        stats = get_feedback_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
