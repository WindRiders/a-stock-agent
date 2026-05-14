# A-Stock Agent Docker 镜像
# 
# 构建:   docker build -t a-stock-agent .
# 运行CLI: docker run --rm a-stock-agent scan
# 运行Web: docker run -p 8501:8501 a-stock-agent web
# 运行API: docker run -p 8000:8000 a-stock-agent api
#
# 环境变量（可选）:
#   -e OPENAI_API_KEY=<key>       LLM 分析
#   -e TELEGRAM_BOT_TOKEN=<token> 通知推送
#   -e DISCORD_WEBHOOK_URL=<url>  通知推送
#
# 持久化数据卷:
#   -v ~/.a-stock-agent:/root/.a-stock-agent

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="A-Stock Agent"
LABEL org.opencontainers.image.description="A股量化交易智能体"
LABEL org.opencontainers.image.source="https://github.com/WindRiders/a-stock-agent"
LABEL org.opencontainers.image.license="MIT"

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /app

# 复制依赖文件
COPY pyproject.toml .
COPY requirements.txt .

# 安装 Python 依赖
# 使用分层缓存：先装核心依赖，再装可选依赖（streamlit/plotly/fastapi）
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir akshare baostock pandas numpy ta \
    && pip install --no-cache-dir matplotlib rich typer pyyaml \
    && pip install --no-cache-dir streamlit plotly fastapi uvicorn

# 复制源码
COPY . .

# 数据卷
VOLUME /root/.a-stock-agent

# 入口点
ENTRYPOINT ["python", "cli.py"]

# 默认命令（无参数显示帮助）
CMD ["--help"]