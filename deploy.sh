#!/bin/bash
# ============================================
# 企业知识库问答 Agent v3.0 - 一键部署脚本
# 域名: agent.wenhuichen.cn
# 服务器: 170.106.158.12
# 项目路径: /opt/agent_rag_project
# ============================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_DIR="/opt/agent_rag_project"
VENV_DIR="$PROJECT_DIR/venv"
DOMAIN="agent.wenhuichen.cn"

echo "============================================"
echo "  企业知识库问答 Agent v3.0 部署"
echo "  域名: $DOMAIN"
echo "============================================"
echo ""

# ===== 1. 系统依赖 =====
log_info "步骤 1/8: 检查系统依赖..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv nginx certbot python3-certbot-nginx

# ===== 2. 项目目录 =====
log_info "步骤 2/8: 准备项目目录..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 如果从 git 部署
if [ -d ".git" ]; then
    log_info "从 Git 拉取最新代码..."
    git pull origin main
fi

# ===== 3. Python 虚拟环境 =====
log_info "步骤 3/8: 配置 Python 环境..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    log_info "虚拟环境已创建"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q
log_info "Python 依赖安装完成"

# ===== 4. 环境变量 =====
log_info "步骤 4/8: 检查环境变量..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    if [ -f "$PROJECT_DIR/backend/.env.example" ]; then
        cp "$PROJECT_DIR/backend/.env.example" "$PROJECT_DIR/.env"
        log_warn ".env 文件已从模板创建，请编辑填入 API Key:"
        log_warn "  vim $PROJECT_DIR/.env"
    else
        log_error "未找到 .env.example 模板！"
        exit 1
    fi
else
    log_info ".env 文件已存在"
fi

# 验证 API Key 是否已配置
if grep -q "your_.*_key_here" "$PROJECT_DIR/.env" 2>/dev/null; then
    log_warn "检测到 .env 中有未修改的占位符，部署后请更新 API Key"
fi

# ===== 5. 创建数据目录 =====
log_info "步骤 5/8: 创建数据目录..."
mkdir -p "$PROJECT_DIR/backend/data"
mkdir -p "$PROJECT_DIR/backend/chroma_store"

# ===== 6. Nginx 配置 =====
log_info "步骤 6/8: 配置 Nginx..."
cp "$PROJECT_DIR/nginx.conf" /etc/nginx/sites-available/agent-wenhuichen
ln -sf /etc/nginx/sites-available/agent-wenhuichen /etc/nginx/sites-enabled/

# 删除默认配置（如果存在）
rm -f /etc/nginx/sites-enabled/default

# 检查 SSL 证书是否存在
if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    log_warn "SSL 证书不存在，先用 HTTP 模式启动 Nginx..."
    # 临时用 HTTP 配置
    cat > /etc/nginx/sites-available/agent-wenhuichen << 'TMPCONF'
server {
    listen 80;
    server_name agent.wenhuichen.cn;
    client_max_body_size 50M;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    location /_stcore/ {
        proxy_pass http://127.0.0.1:8501/_stcore/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
TMPCONF

    nginx -t && systemctl restart nginx
    log_info "尝试申请 SSL 证书..."
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email admin@wenhuichen.cn || {
        log_warn "SSL 证书申请失败，将使用 HTTP 模式运行"
        log_warn "可以稍后手动运行: certbot --nginx -d $DOMAIN"
    }
else
    log_info "SSL 证书已存在"
    # 使用完整的 HTTPS 配置
    cp "$PROJECT_DIR/nginx.conf" /etc/nginx/sites-available/agent-wenhuichen
fi

nginx -t && systemctl reload nginx
log_info "Nginx 配置完成"

# ===== 7. Systemd 服务 =====
log_info "步骤 7/8: 配置系统服务..."

# 停止旧服务
systemctl stop agent-backend 2>/dev/null || true
systemctl stop agent-frontend 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true

# 安装新服务文件
cp "$PROJECT_DIR/agent-backend.service" /etc/systemd/system/
cp "$PROJECT_DIR/agent-frontend.service" /etc/systemd/system/

systemctl daemon-reload
systemctl enable agent-backend agent-frontend
systemctl start agent-backend
sleep 3
systemctl start agent-frontend

log_info "服务已启动"

# ===== 8. 验证 =====
log_info "步骤 8/8: 验证部署..."
echo ""

# 检查后端
if systemctl is-active --quiet agent-backend; then
    log_info "✅ 后端服务: 运行中"
else
    log_error "❌ 后端服务: 启动失败"
    journalctl -u agent-backend --no-pager -n 10
fi

# 检查前端
if systemctl is-active --quiet agent-frontend; then
    log_info "✅ 前端服务: 运行中"
else
    log_error "❌ 前端服务: 启动失败"
    journalctl -u agent-frontend --no-pager -n 10
fi

# 检查 Nginx
if systemctl is-active --quiet nginx; then
    log_info "✅ Nginx: 运行中"
else
    log_error "❌ Nginx: 未运行"
fi

# 等待服务就绪后测试 API
sleep 2
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/ 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    log_info "✅ 后端 API: 正常响应"
else
    log_warn "⚠️  后端 API 暂未响应 (HTTP $HTTP_CODE)，可能还在加载模型..."
fi

echo ""
echo "============================================"
echo "  部署完成！"
echo "============================================"
echo ""
echo "  前端访问:  https://$DOMAIN"
echo "  后端 API:  https://$DOMAIN/api/"
echo "  API 文档:  https://$DOMAIN/api/docs"
echo ""
echo "  管理命令:"
echo "    查看后端日志: journalctl -u agent-backend -f"
echo "    查看前端日志: journalctl -u agent-frontend -f"
echo "    重启后端:     systemctl restart agent-backend"
echo "    重启前端:     systemctl restart agent-frontend"
echo "    重启全部:     systemctl restart agent-backend agent-frontend"
echo ""
echo "  ⚠️  如果还未配置 API Key，请执行:"
echo "    vim $PROJECT_DIR/.env"
echo "    systemctl restart agent-backend"
echo ""
echo "============================================"
