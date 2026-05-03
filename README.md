# Enterprise Knowledge Base Q&A Agent

基于 RAG（检索增强生成）的企业知识库智能问答系统。支持文档上传、向量检索、大模型问答、工具调用和评测面板。

线上地址：https://agent.wenhuichen.cn

## 功能

- **文档上传**：支持 PDF / TXT / DOCX，自动切片、向量化并存入 ChromaDB
- **RAG 问答**：基于上传文档内容进行语义检索 + 大模型生成回答
- **工具调用**：内置时间查询、数学计算等工具，自动识别并调用
- **评测面板**：统计请求数、点赞率、响应时间，可视化趋势图表
- **用户反馈**：支持点赞/点踩，记录到 SQLite 数据库

## 技术架构

```
用户浏览器
    │
    ▼
┌─────────┐     ┌───────────┐     ┌──────────────┐
│ Streamlit│────▶│   Nginx   │────▶│   FastAPI    │
│  前端    │     │ 反向代理  │     │   后端       │
└─────────┘     └───────────┘     └──────┬───────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             ┌──────────┐        ┌──────────┐        ┌──────────────┐
             │ ChromaDB  │        │ 智谱 AI  │        │   SQLite     │
             │ 向量检索  │        │ GLM-4    │        │  反馈存储    │
             └──────────┘        └──────────┘        └──────────────┘
```

## 项目结构

```
├── backend/
│   ├── main.py             # FastAPI 入口，路由定义
│   ├── upload.py           # 文档解析、切片、向量化
│   ├── ask.py              # RAG 检索 + LLM 问答 + 工具调用
│   ├── llm_client.py       # 智谱 AI API 封装
│   ├── tools.py            # 工具函数（时间、计算）
│   ├── feedback.py         # 反馈记录
│   └── db.py               # SQLite 初始化
├── frontend/
│   └── streamlit_app.py    # Streamlit 前端
├── requirements.txt        # Python 依赖
├── nginx.conf              # Nginx 配置
├── agent-backend.service   # systemd 服务文件
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
echo "ZHIPU_API_KEY=your_api_key" > .env
```

获取 API Key：https://open.bigmodel.cn

### 3. 启动服务

```bash
# 启动后端
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# 启动前端
cd frontend && streamlit run streamlit_app.py --server.port 8501
```

### 4. 访问

- 前端：http://localhost:8501
- API 文档：http://localhost:8000/docs

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传文档，自动向量化 |
| `/ask` | POST | 提问，返回 RAG 生成的回答 |
| `/feedback` | POST | 提交用户反馈 |
| `/stats` | GET | 获取统计数据 |

## 生产部署

```bash
# 配置 Nginx
sudo cp nginx.conf /etc/nginx/sites-available/agent
sudo ln -s /etc/nginx/sites-available/agent /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

# 配置 systemd
sudo cp agent-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agent-backend
sudo systemctl start agent-backend

# 启动前端
nohup streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
```

## 依赖

- FastAPI + Uvicorn
- Streamlit + Plotly
- ChromaDB（向量数据库）
- Sentence-Transformers（嵌入模型：all-MiniLM-L6-v2）
- 智谱 AI GLM-4 API
- PyPDF2 + python-docx（文档解析）
- SQLite（反馈存储）
