"""
数据库初始化模块
"""
import os
from feedback import init_feedback_db


def init_db():
    """初始化所有数据库"""
    # 初始化反馈数据库
    init_feedback_db()

    # 确保数据目录存在
    os.makedirs("data", exist_ok=True)
    os.makedirs("chroma_store", exist_ok=True)

    print("✅ 数据库初始化完成")
