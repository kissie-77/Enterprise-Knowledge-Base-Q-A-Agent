# 企业知识库问答 Agent

基于 RAG (Retrieval-Augmented Generation) 的智能问答系统，支持文档上传、向量检索、工具调用和评测面板。

## ✨ 功能特性

- 📄 **文档上传**：支持 PDF/TXT 文件，自动切片并存入向量数据库
- 🔍 **RAG 问答**：基于文档内容进行智能问答，返回相关上下文
- 🛠️ **工具调用**：支持时间查询和数学计算
- 📊 **评测面板**：记录用户反馈，统计准确率、响应时间等指标
- 🚀 **易于部署**：提供完整的部署配置文件

## 🏗️ 项目结构

```
agent_rag_project/
├── backend/                    # FastAPI 后端
│   ├── main.py                 # 主入口
│   ├── upload.py               # 文档上传与向量化
│   ├── ask.py                  # RAG + Agent 问答
│   ├── feedback.py             # 反馈记录
│   ├── db.py                   # 数据库初始化
│   ├── tools.py                # 工具函数
│   ├── llm_client.py           # 智谱 API 封装
│   └── requirements.txt        # 后端依赖
├── frontend/                   # Streamlit 前端
│   └── streamlit_app.py        # 前端应用
├── scripts/                    # 脚本
│   └── init_chroma.py          # 初始化向量库
├── data/                       # 上传文件目录
├── chroma_store/               # Chroma 持久化目录
├── .env.example                # 环境变量模板
├── requirements.txt            # 完整依赖
├── start.sh                    # 启动脚本
├── agent-backend.service       # systemd 服务文件
├── nginx.conf                  # Nginx 配置
└── README.md                   # 本文档
```

## 📋 环境要求

- Python 3.8+
- Linux 服务器（推荐 Ubuntu 20.04+）
- 2GB+ 内存
- 10GB+ 磁盘空间

## 🔧 快速开始

### 1. 获取智谱 AI API Key

1. 访问 [智谱 AI 开放平台](https://open.bigmodel.cn)
2. 注册账号并登录
3. 进入 "API Key" 页面，创建新的 API Key
4. 复制 API Key

### 2. 安装依赖

```bash
# 克隆项目（或上传代码到服务器）
cd /opt
git clone <your-repo-url> agent_rag_project
cd agent_rag_project

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入 API Key
vim .env
```

修改 `.env` 文件：
```bash
ZHIPU_API_KEY=your_actual_api_key_here
```

### 4. 启动服务

#### 方式一：使用启动脚本（推荐）

```bash
chmod +x start.sh
./start.sh
```

#### 方式二：手动启动

**启动后端（FastAPI）：**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

**启动前端（Streamlit）：**
```bash
cd frontend
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### 5. 访问应用

- **前端界面**：`http://your-server-ip:8501`
- **后端 API**：`http://your-server-ip:8000`
- **API 文档**：`http://your-server-ip:8000/docs`

## 🚀 生产部署

### 使用 systemd 服务

1. 复制服务文件：
```bash
sudo cp agent-backend.service /etc/systemd/system/
```

2. 编辑服务文件，修改路径和 API Key：
```bash
sudo vim /etc/systemd/system/agent-backend.service
```

3. 启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable agent-backend
sudo systemctl start agent-backend
sudo systemctl status agent-backend
```

### 使用 Nginx 反向代理

1. 安装 Nginx：
```bash
sudo apt update
sudo apt install nginx
```

2. 配置 Nginx：
```bash
sudo cp nginx.conf /etc/nginx/sites-available/agent-rag
sudo ln -s /etc/nginx/sites-available/agent-rag /etc/nginx/sites-enabled/
```

3. 修改配置中的域名：
```bash
sudo vim /etc/nginx/sites-available/agent-rag
# 将 your-domain.com 替换为你的域名或 IP
```

4. 测试并重启 Nginx：
```bash
sudo nginx -t
sudo systemctl restart nginx
```

### 部署 Streamlit 前端

Streamlit 前端也可以通过 Nginx 代理，或者使用 Streamlit 的生产部署方式：

```bash
# 安装 streamlit
pip install streamlit

# 启动（后台运行）
nohup streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

## 📊 API 接口说明

### 1. 上传文档

**接口**：`POST /upload`

**参数**：
- `file`: 上传的文件（PDF/TXT）
- `chunk_size`: 切片大小（默认 500）
- `overlap`: 重叠字符数（默认 50）

**示例**：
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "chunk_size=500" \
  -F "overlap=50"
```

### 2. 问答

**接口**：`POST /ask`

**参数**：
```json
{
  "question": "你的问题"
}
```

**示例**：
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是 RAG？"}'
```

### 3. 提交反馈

**接口**：`POST /feedback`

**参数**：
```json
{
  "question": "用户问题",
  "answer": "模型回答",
  "feedback": "upvote/downvote",
  "response_time_ms": 100,
  "retrieved_chunks": 4
}
```

### 4. 获取统计信息

**接口**：`GET /stats`

## 🔧 工具调用

系统内置两个工具：

### 1. 获取当前时间

问题示例：
- "现在几点？"
- "当前时间是什么？"

### 2. 数学计算

问题示例：
- "计算 25 * 4"
- "2 + 3 * 4 等于多少？"

## 📈 评测面板

评测面板提供以下统计指标：
- 总请求数
- 点赞率
- 平均响应时间
- 近7天趋势图
- 最近反馈记录

## 🔒 安全配置

### API Key 管理
- API Key 存储在 `.env` 文件中
- 不要将 `.env` 文件提交到版本控制
- 使用 `.gitignore` 排除 `.env` 文件

### 限流配置
- 上传接口：每分钟 10 次
- 问答接口：每分钟 30 次
- 反馈接口：每分钟 60 次

## 🐛 常见问题

### 1. 嵌入模型下载失败

如果下载 `all-MiniLM-L6-v2` 模型失败，可以手动下载：
```bash
# 手动下载模型
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 2. Chroma 数据库权限问题

确保 `chroma_store` 目录有写入权限：
```bash
chmod -R 755 chroma_store
```

### 3. 端口被占用

修改端口号：
- 后端：修改 `main.py` 中的端口
- 前端：修改 `streamlit_app.py` 中的端口

## 📝 开发说明

### 添加新工具

1. 在 `backend/tools.py` 中添加工具函数
2. 在 `backend/ask.py` 中更新 `needs_tool_call` 函数
3. 在 `backend/ask.py` 中更新 `execute_tool` 函数

### 修改嵌入模型

编辑 `backend/upload.py` 中的模型名称：
```python
embedding_model = SentenceTransformer('your-model-name')
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：本项目使用智谱 AI 的免费 API (`glm-4-flash`)，请确保遵守智谱 AI 的使用条款。
