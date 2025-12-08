# Docker 部署指南

## 快速开始

### 1. 生成访问密码

首先需要创建 Basic Auth 密码文件：

```bash
# 使用 Docker 生成密码（推荐）
docker run --rm httpd:alpine htpasswd -nb 用户名 密码 > nginx/.htpasswd

# 或使用系统 htpasswd 命令
# htpasswd -c nginx/.htpasswd 用户名
```

### 2. 配置环境变量（可选）

创建 `.env` 文件：

```bash
cat > .env << EOF
# Jupyter Token（可留空，因为有 nginx 保护）
JUPYTER_TOKEN=your-jupyter-token

# 服务暴露端口
HOST_PORT=8080
EOF
```

### 3. 启动服务

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 4. 访问服务

打开浏览器访问：`http://服务器IP:8080`

- 首先输入 Basic Auth 用户名密码（nginx 层）
- 然后如果设置了 JUPYTER_TOKEN，还需输入 token

## 安全架构

```
用户 → Nginx (Basic Auth) → Jupyter (Token)
     [第一层认证]          [第二层认证]
```

- **Nginx Basic Auth**：必须配置，提供第一层保护
- **Jupyter Token**：可选，提供第二层保护
- **端口隔离**：Jupyter 不直接暴露，只通过 nginx 代理访问

## 常用命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f jupyter
docker-compose logs -f nginx

# 重建镜像
docker-compose build --no-cache

# 重启服务
docker-compose restart
```

## 数据持久化

以下目录会挂载到宿主机，数据不会丢失：

| 容器目录 | 宿主机目录 | 说明 |
|---------|-----------|------|
| /app/data | ./data | 数据文件 |
| /app/output | ./output | 输出文件 |
| /app/notebooks | ./notebooks | 笔记本文件 |

## 修改密码

```bash
# 修改已有用户密码
docker run --rm httpd:alpine htpasswd -nb 用户名 新密码 > nginx/.htpasswd

# 添加新用户
docker run --rm httpd:alpine htpasswd -nb 新用户 密码 >> nginx/.htpasswd

# 重启 nginx 使配置生效
docker-compose restart nginx
```



## 常见问题

### Q: WebSocket 连接失败？
检查 nginx 配置中的 WebSocket 代理设置，确保包含：
```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```

### Q: Kernel 执行超时？
nginx 配置中已设置 API 端点 24 小时超时，如仍有问题可调整 `proxy_read_timeout`。

### Q: 如何更新代码？
```bash
# 停止服务
docker-compose down

# 更新代码后重建
docker-compose build --no-cache
docker-compose up -d
```


## 文件结构

```
binance-rate/
├── Dockerfile              # Docker 镜像构建文件
├── docker-compose.yml      # Docker Compose 配置
├── .dockerignore           # Docker 构建排除文件
├── .env                    # 环境变量（需自己创建）
├── nginx/
│   ├── nginx.conf          # Nginx 配置
│   ├── .htpasswd           # 密码文件（需自己创建）
│   └── .htpasswd.example   # 密码文件示例
├── requirements.txt        # Python 依赖
├── *.ipynb                 # Jupyter Notebooks
└── *.py                    # Python 模块
```
