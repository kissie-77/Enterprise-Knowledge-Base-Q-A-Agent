#!/bin/bash
# 启动脚本 - 企业知识库问答 Agent

set -e

# 项目根目录
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_DIR"

echo "🚀 启动企业知识库问答 Agent..."

# 检查虚拟环境
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 检查并安装依赖
echo "📦 检查依赖..."
pip install -r requirements.txt -q

# 检查环境变量
if [ ! -f ".env" ]; then
    echo "⚠️  警告: 未找到 .env 文件"
    echo "请复制 .env.example 为 .env 并配置 API Key"
    exit 1
fi

# 加载环境变量
export $(grep -v '^#' .env | xargs)

# 检查 API Key
if [ -z "$ZHIPU_API_KEY" ] || [ "$ZHIPU_API_KEY" = "your_zhipu_api_key_here" ]; then
    echo "⚠️  警告: ZHIPU_API_KEY 未配置"
    echo "请在 .env 文件中配置智谱 AI API Key"
    exit 1
fi

# 启动后端服务
echo "🔄 启动后端服务 (FastAPI)..."
cd "$PROJECT_DIR/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2 &
BACKEND_PID=$!
echo "✅ 后端服务已启动 (PID: $BACKEND_PID)"

# 等待后端服务启动
sleep 3

# 启动前端服务
echo "🔄 启动前端服务 (Streamlit)..."
cd "$PROJECT_DIR/frontend"
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "✅ 前端服务已启动 (PID: $FRONTEND_PID)"

echo ""
echo "🎉 服务启动成功！"
echo ""
echo "访问地址:"
echo "  - 前端 (Streamlit): http://localhost:8501"
echo "  - 后端 API (FastAPI): http://localhost:8000"
echo "  - API 文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"

# 等待用户中断
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
