#!/bin/bash
# 服务器设置脚本 - 添加公钥并部署

set -e

echo "========================================"
echo "服务器设置脚本"
echo "========================================"

# 1. 添加公钥到 authorized_keys
echo "1. 添加公钥..."
mkdir -p ~/.ssh
cat >> ~/.ssh/authorized_keys << 'EOF'
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDfTpGfWoyJB8g4Zuuf5NfJGVCG2iPyYvle7zNj/41AlxrncsCszUVZWvqZBE+x6yDPDhO94ZvSIveKF7+P9aK6eETyRnFE3Lzycl1GVNLELMc7S73eMqEmWMXLBN+4pgumDrSFwTLSdhFLJsH0L0ycgEAl1a6/yov+tSu7sJ1nxuQHv+YHes4uixdGhD1vz+kGkX60cvw672vzZ9ETyDTEq8oBM9BLiCChSEqw0GTSe0XPrXUKjFejDsM0VwjwY1UF/9pGTY6zjnbPhoytwjmYX4WQJgjcTigVO1JJOxk4IdtL0SeqThyCxlMGUL01pIBFbb+hlMSteXKzf6GJ0VK9 skey-fz8najcp
EOF

# 2. 设置权限
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

echo "✓ 公钥添加完成"

# 3. 检查项目目录
if [ ! -d "/opt/agent_rag_project" ]; then
    echo "错误: 项目目录不存在 /opt/agent_rag_project"
    echo "请先上传代码到 /opt/agent_rag_project"
    exit 1
fi

# 4. 执行部署
cd /opt/agent_rag_project
chmod +x deploy.sh
./deploy.sh
