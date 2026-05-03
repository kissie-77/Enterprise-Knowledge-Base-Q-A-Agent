# 手动部署指南

由于 SSH 密钥认证问题，请按以下步骤手动部署。

## 步骤 1: 上传代码到服务器

### 方法 A: 使用 WinSCP（推荐）
1. 下载并安装 [WinSCP](https://winscp.net/)
2. 连接到服务器：
   - 主机: `170.106.158.12`
   - 端口: `22`
   - 用户名: `root`
   - 密码: `1159633cwhabc`
3. 上传整个项目文件夹到 `/opt/agent_rag_project`

### 方法 B: 使用命令行
```bash
# 在本地 PowerShell 或 CMD 中执行
scp -r "D:\Workspace\Enterprise Knowledge Base Q&A Agent\*" root@170.106.158.12:/opt/agent_rag_project/
```

## 步骤 2: 在服务器上执行部署脚本

登录服务器并执行部署脚本：

```bash
# 登录服务器
ssh root@170.106.158.12

# 进入项目目录
cd /opt/agent_rag_project

# 设置脚本权限
chmod +x deploy.sh

# 执行部署脚本
./deploy.sh
```

## 步骤 3: 配置环境变量

编辑 `.env` 文件，填入智谱 AI API Key：

```bash
vim /opt/agent_rag_project/.env
```

修改这一行：
```bash
ZHIPU_API_KEY=your_actual_api_key_here
```

## 步骤 4: 重启服务

```bash
# 重启后端服务
systemctl restart agent-backend

# 重启前端服务
pkill -f "streamlit run"
cd /opt/agent_rag_project/frontend
source ../venv/bin/activate
nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > ../frontend.log 2>&1 &
```

## 步骤 5: 检查服务状态

```bash
# 检查后端服务
systemctl status agent-backend

# 检查前端进程
ps aux | grep streamlit

# 检查 Nginx
nginx -t
systemctl status nginx
```

## 步骤 6: 测试访问

访问以下地址：
- **前端界面**: https://agent.wenhuichen.cn
- **后端 API 文档**: https://agent.wenhuichen.cn/api/docs

## 常见问题

### 1. 端口被占用
```bash
# 查看端口占用
netstat -tlnp | grep 8000
netstat -tlnp | grep 8501

# 杀死占用进程
kill -9 <PID>
```

### 2. Nginx 配置错误
```bash
# 测试配置
nginx -t

# 查看错误日志
tail -f /var/log/nginx/error.log
```

### 3. 服务启动失败
```bash
# 查看后端日志
journalctl -u agent-backend -f

# 查看前端日志
tail -f /opt/agent_rag_project/frontend.log
```

### 4. 域名解析问题
```bash
# 检查域名解析
nslookup agent.wenhuichen.cn

# 检查防火墙
ufw status
```

## 卸载/重新部署

```bash
# 停止服务
systemctl stop agent-backend
pkill -f "streamlit run"

# 删除旧文件
rm -rf /opt/agent_rag_project

# 重新上传代码并执行部署脚本
```
