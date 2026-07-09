#将cuda,换成cpu如果不支持cuda

# 选择 Python 基础镜像
FROM python:3.13-slim

#安装ffmpeg在image内
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装 1.复制list 2.安装list里的依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目
COPY . .

# Docker Desktop 会显示端口
EXPOSE 8001

# 进入目录后运行应用
WORKDIR /app/server
CMD ["python", "server-rt-stt.py"]

# 开始build //CMD里运行
# docker build -t open-ai-whisper .

# 如果要build 不同架构cpu打混合image
# docker buildx build --platform linux/amd64,linux/arm64 -t username/target_repository:v1.0(version)orlastest --push(push to hub) .

# 开启container运行
# docker run -it/-d(-it进入cli, -d后台运行) --name open-ai-whisper -p 8001:8001 -v ~/.cache:/root/.cache rouben99/open-ai-whisper:latest