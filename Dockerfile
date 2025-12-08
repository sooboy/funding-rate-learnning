FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY *.ipynb .
COPY *.py .

# 创建数据目录
RUN mkdir -p /app/data /app/output

# 暴露端口
EXPOSE 8888

# 设置环境变量默认值
ENV JUPYTER_TOKEN=""
ENV JUPYTER_PASSWORD=""

# 启动 Jupyter Lab
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=${JUPYTER_TOKEN} --NotebookApp.password=${JUPYTER_PASSWORD}"]
