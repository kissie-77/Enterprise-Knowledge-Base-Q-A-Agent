"""
初始化 Chroma 向量数据库
"""
import sys
import os

# 添加 backend 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from upload import get_chroma_client, COLLECTION_NAME


def init_chroma():
    """初始化 Chroma 数据库"""
    client = get_chroma_client()

    # 创建或获取集合
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 清空现有数据（可选）
    # collection.delete()

    print(f"✅ Chroma 数据库初始化完成")
    print(f"   集合名称: {COLLECTION_NAME}")
    print(f"   存储路径: chroma_store/")


if __name__ == "__main__":
    init_chroma()
