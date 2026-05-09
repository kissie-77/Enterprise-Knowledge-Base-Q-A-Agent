# 部署手册 - 企业知识库问答 Agent v3.0

> 部署域名: `agent.wenhuichen.cn`  
> 服务器: `170.106.158.12`  
> 项目路径: `/opt/agent_rag_project`

---

## 快速部署（一键脚本）

```bash
# 1. 登录服务器
ssh root@170.106.158.12

# 2. 进入项目目录（如果是首次部署，先上传代码）
cd /opt/agent_rag_project

# 3. 执行一键部署
chmod +x deploy.sh
./deploy.sh

# 4. 配置 API Key
vim .env
# 至少填入一个 LLM API Key，如：
# LLM_PROVIDER=zhipu
# ZHIPU_API_KEY=your_actual_key

# 5. 重启后端使配置生效
systemctl restart agent-backend
```

---

## 手动分步部署

### 步骤 1: 上传代码

**方法 A: Git 拉取（推荐）**
```bash
ssh root@170.106.158.12
cd /opt
git clone https://github.com/kissie-77/Enterprise-Knowledge-Base-Q-A-Agent.git agent_rag_project
cd agent_rag_project
git checkout feat/hello-agents-architecture
```

**方法 B: SCP 上传**
```bash
scp -r ./* root@170.106.158.12:/opt/agent_rag_project/
```

### 步骤 2: 安装 Python 环境

```bash
cd /opt/agent_rag_project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 步骤 3: 配置环境变量

```bash
cp backend/.env.example .env
vim .env
```

**最低配置**（使用智谱免费模型）：
```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=你的智谱API密钥

# 可选：关闭某些高级模块以加快响应
ENABLE_HYDE=true
ENABLE_RERANKER=true
REFLECTION_MAX_ITER=2
REFLECTION_THRESHOLD=7.0
COMPRESSOR_WINDOW=6
```

### 步骤 4: 创建数据目录

```bash
mkdir -p backend/data
mkdir -p backend/chroma_store
```

### 步骤 5: 配置 Nginx

```bash
cp nginx.conf /etc/nginx/sites-available/agent-wenhuichen
ln -sf /etc/nginx/sites-available/agent-wenhuichen /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
```

**SSL 证书**（如果还没有）：
```bash
apt install certbot python3-certbot-nginx
certbot --nginx -d agent.wenhuichen.cn
```

### 步骤 6: 配置 Systemd 服务

```bash
# 安装服务文件
cp agent-backend.service /etc/systemd/system/
cp agent-frontend.service /etc/systemd/system/

# 启用并启动
systemctl daemon-reload
systemctl enable agent-backend agent-frontend
systemctl start agent-backend
sleep 3
systemctl start agent-frontend
```

### 步骤 7: 验证

```bash
# 检查服务状态
systemctl status agent-backend
systemctl status agent-frontend

# 测试 API
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health

# 测试外部访问
curl https://agent.wenhuichen.cn/api/
```

---

## 日常运维

### 查看日志

```bash
# 后端日志（实时）
journalctl -u agent-backend -f

# 前端日志
journalctl -u agent-frontend -f

# Nginx 日志
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### 重启服务

```bash
# 重启后端（修改代码或配置后）
systemctl restart agent-backend

# 重启前端
systemctl restart agent-frontend

# 全部重启
systemctl restart agent-backend agent-frontend nginx
```

### 更新代码

```bash
cd /opt/agent_rag_project
git pull origin feat/hello-agents-architecture

# 如果有新依赖
source venv/bin/activate
pip install -r requirements.txt

# 重启服务
systemctl restart agent-backend agent-frontend
```

### 查看知识库状态

```bash
curl https://agent.wenhuichen.cn/api/stats
```

---

## 架构说明

```
用户浏览器
    │
    ▼ HTTPS (443)
┌──────────┐
│  Nginx   │  agent.wenhuichen.cn
│  反向代理 │
└──┬───┬───┘
   │   │
   │   │  /api/* → 127.0.0.1:8000
   │   ▼
   │  ┌──────────────┐
   │  │ FastAPI 后端  │  agent-backend.service
   │  │ (uvicorn)    │  WorkingDirectory: /opt/agent_rag_project/backend
   │  └──────────────┘
   │
   │  /* → 127.0.0.1:8501
   ▼
┌──────────────┐
│ Streamlit    │  agent-frontend.service
│ 前端         │  WorkingDirectory: /opt/agent_rag_project/frontend
└──────────────┘
```

### 端口使用

| 端口 | 服务 | 说明 |
|------|------|------|
| 80 | Nginx | HTTP → 301 重定向到 HTTPS |
| 443 | Nginx | HTTPS 入口 |
| 8000 | FastAPI | 后端 API（仅监听 127.0.0.1） |
| 8501 | Streamlit | 前端界面（仅监听 127.0.0.1） |

### 文件结构（服务器上）

```
/opt/agent_rag_project/
├── .env                    # 环境变量（API Key 等）
├── requirements.txt
├── nginx.conf
├── agent-backend.service
├── agent-frontend.service
├── deploy.sh
├── venv/                   # Python 虚拟环境
├── backend/
│   ├── main.py             # 后端入口
│   ├── core/               # 核心模块
│   ├── data/               # 上传的文档
│   └── chroma_store/       # 向量数据库持久化
└── frontend/
    └── streamlit_app.py    # 前端入口
```

---

## 常见问题

### 1. 后端启动失败

```bash
# 查看详细错误
journalctl -u agent-backend -n 50 --no-pager

# 常见原因：
# - .env 中 API Key 未配置 → 服务仍会启动，但 LLM 不可用
# - 端口被占用 → netstat -tlnp | grep 8000
# - 依赖缺失 → source venv/bin/activate && pip install -r requirements.txt
```

### 2. 前端无法加载

```bash
# 确认 Streamlit 进程存在
systemctl status agent-frontend

# 确认端口监听
ss -tlnp | grep 8501

# 检查 Nginx 代理
curl -I http://127.0.0.1:8501
```

### 3. 上传文件失败 (413 错误)

Nginx 的 `client_max_body_size` 已设为 50M。如需更大：
```bash
vim /etc/nginx/sites-available/agent-wenhuichen
# 修改 client_max_body_size 100M;
systemctl reload nginx
```

### 4. API 超时 (504 错误)

启用了反思模块时响应较慢（需要多次 LLM 调用）：
```bash
# 方案1: 关闭反思（前端请求中 use_reflection=false）
# 方案2: 加大超时（nginx.conf 中已设为 120s）
# 方案3: 使用更快的模型（如 glm-4-flash）
```

### 5. SSL 证书续期

Let's Encrypt 证书 90 天过期，certbot 自动续期：
```bash
# 测试续期
certbot renew --dry-run

# 手动续期
certbot renew
systemctl reload nginx
```

### 6. 磁盘空间不足

```bash
# 检查向量数据库大小
du -sh /opt/agent_rag_project/backend/chroma_store/

# 清理上传的原始文件（向量已存储，可删除原文件）
rm -rf /opt/agent_rag_project/backend/data/*

# 清理日志
journalctl --vacuum-time=7d
```

---

## 性能调优

### 加快首次响应

嵌入模型首次加载较慢（约 10-30 秒），之后会缓存。建议：
```bash
# 部署后手动触发一次加载
curl -X POST https://agent.wenhuichen.cn/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "测试", "use_reflection": false}'
```

### 关闭不需要的模块

在 `.env` 中调整：
```env
# 关闭 HyDE（减少一次 LLM 调用）
ENABLE_HYDE=false

# 关闭 Reranker（减少精排耗时）
ENABLE_RERANKER=false

# 减少反思迭代次数
REFLECTION_MAX_ITER=1
```

### 增加 Worker 数量

修改 `agent-backend.service` 中的 `--workers` 参数：
```
ExecStart=... --workers 4
```

然后 `systemctl daemon-reload && systemctl restart agent-backend`
