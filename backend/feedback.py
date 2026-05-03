"""
反馈记录与统计模块
"""
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List


DB_PATH = "feedback.db"


def init_feedback_db():
    """初始化反馈数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建反馈表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            feedback TEXT NOT NULL,
            response_time_ms INTEGER,
            retrieved_chunks INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


def save_feedback(
    question: str,
    answer: str,
    feedback: str,
    response_time_ms: int = 0,
    retrieved_chunks: int = 0,
    question_id: str = None
):
    """
    保存用户反馈

    Args:
        question: 用户问题
        answer: 模型回答
        feedback: 反馈类型（upvote/downvote）
        response_time_ms: 响应时间（毫秒）
        retrieved_chunks: 检索到的片段数量
        question_id: 问题ID（可选）
    """
    init_feedback_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO feedback (question_id, question, answer, feedback, response_time_ms, retrieved_chunks)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (question_id, question, answer, feedback, response_time_ms, retrieved_chunks))

    conn.commit()
    conn.close()


def get_feedback_stats() -> Dict[str, Any]:
    """
    获取反馈统计信息

    Returns:
        包含各种统计指标的字典
    """
    init_feedback_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 总请求数
    cursor.execute("SELECT COUNT(*) as count FROM feedback")
    total_requests = cursor.fetchone()["count"]

    # 点赞数
    cursor.execute("SELECT COUNT(*) as count FROM feedback WHERE feedback = 'upvote'")
    upvotes = cursor.fetchone()["count"]

    # 点踩数
    cursor.execute("SELECT COUNT(*) as count FROM feedback WHERE feedback = 'downvote'")
    downvotes = cursor.fetchone()["count"]

    # 平均响应时间
    cursor.execute("SELECT AVG(response_time_ms) as avg_time FROM feedback WHERE response_time_ms > 0")
    avg_response_time = cursor.fetchone()["avg_time"]
    avg_response_time = round(avg_response_time, 2) if avg_response_time else 0

    # 点赞率
    total_feedback = upvotes + downvotes
    upvote_rate = (upvotes / total_feedback * 100) if total_feedback > 0 else 0

    # 近7天趋势
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    cursor.execute('''
        SELECT DATE(created_at) as date,
               COUNT(*) as count,
               SUM(CASE WHEN feedback = 'upvote' THEN 1 ELSE 0 END) as upvotes,
               SUM(CASE WHEN feedback = 'downvote' THEN 1 ELSE 0 END) as downvotes
        FROM feedback
        WHERE DATE(created_at) >= ?
        GROUP BY DATE(created_at)
        ORDER BY date
    ''', (seven_days_ago,))

    daily_stats = []
    for row in cursor.fetchall():
        daily_stats.append({
            "date": row["date"],
            "count": row["count"],
            "upvotes": row["upvotes"],
            "downvotes": row["downvotes"]
        })

    # 最近的反馈记录
    cursor.execute('''
        SELECT * FROM feedback
        ORDER BY created_at DESC
        LIMIT 10
    ''')
    recent_feedback = []
    for row in cursor.fetchall():
        recent_feedback.append({
            "id": row["id"],
            "question": row["question"],
            "feedback": row["feedback"],
            "created_at": row["created_at"]
        })

    conn.close()

    return {
        "total_requests": total_requests,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "upvote_rate": round(upvote_rate, 2),
        "avg_response_time_ms": avg_response_time,
        "daily_stats": daily_stats,
        "recent_feedback": recent_feedback
    }


def get_recent_feedback(limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取最近的反馈记录

    Args:
        limit: 返回记录数量

    Returns:
        反馈记录列表
    """
    init_feedback_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM feedback
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))

    results = []
    for row in cursor.fetchall():
        results.append(dict(row))

    conn.close()
    return results
