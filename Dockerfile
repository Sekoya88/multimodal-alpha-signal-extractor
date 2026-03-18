FROM python:3.11-slim

# 1. Install system dependencies for build and Ollama
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/.ollama/models

WORKDIR $HOME/app

# Create dir for Ollama models to avoid permission issues
RUN mkdir -p $OLLAMA_MODELS

# 4. Copy everything (we need README and src/ for the editable install)
COPY --chown=user:user . .

# 5. Install the project and dependencies
# We use the pre-built wheels for llama-cpp-python to avoid compiling C++ for 15 minutes and timing out.
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir -e .

# 7. Make the entrypoint executable
RUN chmod +x scripts/entrypoint.sh

# HF Spaces exposes port 7860
EXPOSE 7860

# 8. Start everything
ENTRYPOINT ["./scripts/entrypoint.sh"]
