"""
部署脚本 - 上传代码到服务器并配置
"""
import paramiko
import os
import sys

# 服务器配置
HOST = "170.106.158.12"
PORT = 22
USER = "root"
PASSWORD = "1159633cwhabc"
REMOTE_DIR = "/opt/agent_rag_project"

# 本地项目目录
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

def create_ssh_client():
    """创建 SSH 连接"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    return client

def upload_directory(sftp, local_path, remote_path):
    """递归上传目录"""
    # 创建远程目录
    try:
        sftp.mkdir(remote_path)
    except IOError:
        pass  # 目录已存在

    # 遍历本地目录
    for item in os.listdir(local_path):
        local_item = os.path.join(local_path, item)
        remote_item = os.path.join(remote_path, item)

        if os.path.isdir(local_item):
            # 递归上传子目录
            upload_directory(sftp, local_item, remote_item)
        else:
            # 上传文件
            try:
                sftp.put(local_item, remote_item)
                print(f"✓ 上传: {item}")
            except Exception as e:
                print(f"✗ 上传失败 {item}: {e}")

def run_commands(client, commands):
    """在服务器上执行命令"""
    for cmd in commands:
        print(f"执行: {cmd}")
        stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
        exit_code = stdout.channel.recv_exit_status()
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')

        if exit_code != 0:
            print(f"错误: {error}")
        else:
            if output:
                print(output.strip())
        print("-" * 50)

def main():
    print("=" * 60)
    print("开始部署到服务器...")
    print(f"服务器: {USER}@{HOST}")
    print("=" * 60)

    try:
        # 创建 SSH 连接
        print("\n1. 连接服务器...")
        client = create_ssh_client()
        sftp = client.open_sftp()
        print("✓ 连接成功")

        # 创建远程目录
        print("\n2. 创建远程目录...")
        try:
            sftp.mkdir(REMOTE_DIR)
        except IOError:
            print("目录已存在")
        print("✓ 目录准备完成")

        # 上传代码
        print("\n3. 上传代码...")
        upload_directory(sftp, LOCAL_DIR, REMOTE_DIR)
        print("✓ 代码上传完成")

        # 关闭 SFTP
        sftp.close()

        # 在服务器上执行配置命令
        print("\n4. 配置服务器环境...")

        commands = [
            # 安装系统依赖
            "apt-get update",
            "apt-get install -y python3 python3-pip python3-venv nginx",

            # 进入项目目录
            f"cd {REMOTE_DIR}",

            # 创建虚拟环境
            "python3 -m venv venv",

            # 安装 Python 依赖
            f"cd {REMOTE_DIR} && source venv/bin/activate && pip install -r requirements.txt",

            # 配置 Nginx
            f"cp {REMOTE_DIR}/nginx.conf /etc/nginx/sites-available/agent-wenhuichen",
            "ln -sf /etc/nginx/sites-available/agent-wenhuichen /etc/nginx/sites-enabled/",
            "nginx -t",
            "systemctl restart nginx",

            # 配置 systemd 服务
            f"cp {REMOTE_DIR}/agent-backend.service /etc/systemd/system/",
            "systemctl daemon-reload",
            "systemctl enable agent-backend",
            "systemctl start agent-backend",

            # 启动前端（后台运行）
            f"cd {REMOTE_DIR}/frontend && source ../venv/bin/activate && nohup streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > {REMOTE_DIR}/frontend.log 2>&1 &",
        ]

        run_commands(client, commands)

        # 设置文件权限
        print("\n5. 设置文件权限...")
        permissions_commands = [
            f"chmod +x {REMOTE_DIR}/start.sh",
            f"chmod -R 755 {REMOTE_DIR}",
        ]
        run_commands(client, permissions_commands)

        # 检查服务状态
        print("\n6. 检查服务状态...")
        status_commands = [
            "systemctl status agent-backend --no-pager",
            "ps aux | grep streamlit",
            "nginx -t",
        ]
        run_commands(client, status_commands)

        # 关闭连接
        client.close()

        print("\n" + "=" * 60)
        print("部署完成！")
        print("=" * 60)
        print(f"前端访问: https://agent.wenhuichen.cn")
        print(f"后端 API: https://agent.wenhuichen.cn/api/docs")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 部署失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
