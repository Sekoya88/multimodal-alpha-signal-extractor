#!/bin/bash
# Entrypoint for HF Spaces Docker deployment.
# Local dev: run `streamlit run app/streamlit_app.py` directly (this script is NOT used).

set -e

APP_DIR="${HOME}/app"
VLM_MODEL="alpha-signal-q4km.gguf"
VLM_PATH="${APP_DIR}/${VLM_MODEL}"
MMPROJ_DIR="${APP_DIR}/models"
MMPROJ_FILE="mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
MMPROJ_PATH="${MMPROJ_DIR}/${MMPROJ_FILE}"

# --- 1. Start Ollama Server in Background ---
echo "⚙️  Starting Ollama Server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama API to be reachable
echo "⏳ Waiting for Ollama to be ready..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done
echo "✅ Ollama is up!"

# --- 2. Pull the Text Sentiment Model ---
# Override via OLLAMA_MODEL env (e.g. OLLAMA_MODEL=qwen2.5:0.5b for free-tier speed).
MODEL_NAME="${OLLAMA_MODEL:-llama3:8b}"
echo "⬇️  Pulling $MODEL_NAME (this may take a few minutes the first time)..."
ollama pull $MODEL_NAME
echo "✅ $MODEL_NAME is ready!"

# Warm up: pre-load model via /api/chat (same as LangChain) to avoid 500 on first user request
echo "🔥 Warming up sentiment model..."
curl -s -X POST http://localhost:11434/api/chat -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL_NAME"'","messages":[{"role":"user","content":"Hi"}],"stream":false}' > /dev/null || true
echo "✅ Model warmed up."

# --- 3. Download VLM (fine-tuned alpha-signal) from HF Hub ---
if [ ! -f "$VLM_PATH" ]; then
    echo "⚠️  $VLM_MODEL not found locally."
    echo "⬇️  Downloading from Sekoya/mon-qwen-finetune..."
    wget -qO "$VLM_PATH" "https://huggingface.co/Sekoya/mon-qwen-finetune/resolve/main/alpha-signal-q4km.gguf"
    echo "✅ VLM model downloaded!"
else
    echo "✅ VLM model already present."
fi

# --- 4. Download mmproj (vision encoder) if missing ---
if [ ! -f "$MMPROJ_PATH" ]; then
    echo "⚠️  $MMPROJ_FILE not found."
    echo "⬇️  Downloading from ggml-org/Qwen2.5-VL-3B-Instruct-GGUF..."
    mkdir -p "$MMPROJ_DIR"
    wget -qO "$MMPROJ_PATH" "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/${MMPROJ_FILE}"
    echo "✅ mmproj downloaded!"
else
    echo "✅ mmproj already present."
fi

# --- 5. Export paths for config.py (env overrides) ---
export LLAMA_CPP_MODEL_PATH="$VLM_PATH"
export LLAMA_CPP_MMPROJ_PATH="$MMPROJ_PATH"

# --- 6. Start Streamlit App ---
echo "🚀 Starting Alpha-Signal Extractor Dashboard on port 7860..."
python -m streamlit run app/streamlit_app.py --server.port=7860 --server.address=0.0.0.0
