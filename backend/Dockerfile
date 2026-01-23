FROM python:3.11-slim

# =========================
# 1️⃣ 换源 + 安装系统级依赖（FFmpeg）
# =========================
RUN rm -f /etc/apt/sources.list && \
    rm -rf /etc/apt/sources.list.d/* && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# =========================
# 2️⃣ 环境变量（🔥 关键）
# =========================
ENV PATH="/usr/bin:${PATH}"
ENV HF_ENDPOINT=https://hf-mirror.com
ENV TRANSCRIBER_TYPE=bcut


# =========================
# 3️⃣ 工作目录
# =========================
WORKDIR /app

# =========================
# 4️⃣ 先拷贝依赖清单（利用 Docker 缓存）
# （你现在 docker build 就是在 backend 目录 👍）
# =========================
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# =========================
# 5️⃣ 再拷贝全部源代码
# =========================
COPY . /app

# =========================
# 6️⃣ 启动入口（CLI / Web 自动分流）
# =========================
CMD ["python", "main.py"]
