#!/bin/bash
# 部署脚本 - 在服务器上执行

set -e

echo "========================================"
echo "企业知识库问答 Agent 部署脚本"
echo "========================================"

# 1. 更新系统
echo "1. 更新系统..."
apt-get update
apt-get install -y python3 python3-pip python3-venv nginx

# 2. 进入项目目录
cd /opt/agent_rag_project

# 3. 创建虚拟环境
echo "2. 创建虚拟环境..."
python3 -m venv venv

# 4. 安装 Python 依赖
echo "3. 安装 Python 依赖..."
source venv/bin/activate
pip install -r requirements.txt

# 5. 配置 Nginx
echo "4. 配置 Nginx..."
cp nginx.conf /etc/nginx/sites-available/agent-wenhuichen
ln -sf /etc/nginx/sites-available/agent-wenhuichen /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# 6. 配置 systemd 服务
echo "5. 配置 systemd 服务..."
cp agent-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable agent-backend
systemctl start agent-backend

# 7. 启动前端（后台运行）
echo "6. 启动前端服务..."
cd frontend
source ../venv/bin/activate
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > ../frontend.log 2>&1 &

# 8. 设置权限
echo "7. 设置文件权限..."
cd /opt/agent_rag_project
chmod +x start.sh
chmod -R 755 .

# 9. 检查服务状态
echo "8. 检查服务状态..."
echo "后端服务状态:"
systemctl status agent-backend --no-pager
echo ""
echo "前端进程:"
ps aux | grep streamlit | grep -v grep
echo ""
echo "Nginx 配置测试:"
nginx -t

echo ""
echo "========================================"
echo "部署完成！"
echo "========================================"
echo "前端访问: https://agent.wenhuichen.cn"
echo "后端 API: https://agent.wenhuichen.cn/api/docs"
echo "========================================"
