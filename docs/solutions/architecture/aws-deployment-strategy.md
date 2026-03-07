---
type: architecture_decision
status: proposed
category: infrastructure
tags: [aws, deployment, docker, ec2, llm]
created_at: 2026-03-07
---

# AWS Deployment Strategy for Multimodal Alpha-Signal Extractor

## 1. Context & Problem
The application needs to be deployed to a production-like environment for testing. 
Unlike standard web applications, this system runs local Large Language Models (LLMs) and Vision-Language Models (VLMs) via `llama.cpp` (Qwen2.5-VL) and `ollama` (LLaMA 3). 
These models require significant memory (RAM) and compute (ideally GPU) to run efficiently. 

## 2. Analysis of Options

### Option A: Serverless Containers (AWS Fargate / App Runner)
*   **Pros**: Easy to deploy, no server management.
*   **Cons**: No GPU support (or very complex/expensive via ECS). Strict memory limits (max 16GB for App Runner, 30GB for Fargate). Loading heavy GGUF models into memory on a serverless container is slow and inefficient per request.

### Option B: Virtual Machines (Amazon EC2) - **RECOMMENDED**
*   **Pros**: Full control over hardware. Can attach GPUs (g4dn/g5 instances) or provision high-RAM CPU instances (t3/c6a). Docker and Docker Compose run natively.
*   **Cons**: Requires managing the OS and Docker daemon.

## 3. Proposed Architecture
*   **Containerization**: A single `Dockerfile` packaging the Python environment, Streamlit, and `llama.cpp` bindings. Models (.gguf) can be baked into the image or downloaded at runtime via an S3 bucket or volume mount.
*   **Compute**: Amazon EC2 instance.
*   **Network**: Security Group exposing port 8501 (Streamlit UI) and 22 (SSH for admin).
*   **Storage**: 50GB Amazon EBS (gp3) to store the OS, Docker images, and local model weights.

## 4. Compute Sizing & Cost Estimates

| Instance Type | Specs | Pros/Cons | Est. Monthly Cost |
| :--- | :--- | :--- | :--- |
| **t3.xlarge** | 4 vCPU, 16GB RAM | **Pros**: Cheap. **Cons**: CPU-only inference (slower generation, ~2-5 tok/s). | ~$120 / month |
| **g4dn.xlarge** | 4 vCPU, 16GB RAM, NVIDIA T4 (16GB VRAM) | **Pros**: Fast inference via GPU acceleration. **Cons**: Expensive for simple testing. | ~$380 / month |

*Storage (50GB gp3)*: ~$4.00 / month.

## 5. Deployment Workflow
1. Write `Dockerfile` and `docker-compose.yml`.
2. Provision EC2 via IaC (Terraform or AWS CDK).
3. Use a startup script (User Data) to install Docker, pull the codebase, and run `docker-compose up -d`.