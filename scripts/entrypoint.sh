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
# you must either upload it to your Space via Git LFS, OR download it here.
# Example fallback download if the file is missing:
VLM_MODEL="alpha-signal-q4km.gguf"
if [ ! -f "$VLM_MODEL" ]; then
    echo "⚠️  $VLM_MODEL not found locally."
    echo "Please upload it using Git LFS to your Hugging Face Space repository."
    echo "Or uncomment the wget line below if you host it elsewhere."
    # wget -qO $VLM_MODEL "https://huggingface.co/your-username/your-model/resolve/main/alpha-signal-q4km.gguf"
fi

# --- 4. Start Streamlit App ---
echo "🚀 Starting Alpha-Signal Extractor Dashboard on port 7860..."
python -m streamlit run app/streamlit_app.py --server.port=7860 --server.address=0.0.0.0
