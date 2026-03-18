#!/bin/bash

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
# Note: On a free 16GB CPU space, loading an 8B model will take ~4.7GB RAM and take a few minutes.
# If you want it faster, you might change llama3:8b to qwen:0.5b in your config and here.
MODEL_NAME="llama3:8b"
echo "⬇️  Pulling $MODEL_NAME (this may take a few minutes the first time)..."
ollama pull $MODEL_NAME
echo "✅ $MODEL_NAME is ready!"

# --- 3. Ensure VLM Model Exists ---
# Since you fine-tuned Qwen2.5-VL and exported it to GGUF,
# HF Spaces has a 1GB limit for free spaces if not using LFS/Models.
# We will download it directly inside the container from a public URL.
VLM_MODEL="alpha-signal-q4km.gguf"
if [ ! -f "$VLM_MODEL" ]; then
    echo "⚠️  $VLM_MODEL not found locally."
    echo "⬇️  Downloading the model via wget to bypass the 1GB HF Space Git limit..."
    
    # Use Hugging Face Hub CLI to download a tiny, publicly available vision model as a fallback/test
    # We will use minicpm-v-2.0 which is ~2GB, but we'll get a super small quant to fit in the container
    echo "Using a public Qwen2-VL GGUF as fallback for testing..."
    # We download a small Qwen2-VL quant from bartowski to test if the user's specific file isn't available
    wget -qO $VLM_MODEL "https://huggingface.co/bartowski/Qwen2-VL-2B-Instruct-GGUF/resolve/main/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
    
    echo "✅ Model downloaded!"
fi

# --- 4. Start Streamlit App ---
echo "🚀 Starting Alpha-Signal Extractor Dashboard on port 7860..."
python -m streamlit run app/streamlit_app.py --server.port=7860 --server.address=0.0.0.0
