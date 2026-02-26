# ==========================================
# Stage 1: Builder (Compile llama.cpp)
# ==========================================
FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /build/llama.cpp

# Build with generic architecture flags (Critical for Cloud Run compatibility)
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=OFF
RUN cmake --build build --config Release -j$(nproc)

# ==========================================
# Stage 2: Runtime (Download Your Model & Run)
# ==========================================
FROM ubuntu:22.04

# Install Python/Pip for Hugging Face CLI & Runtime deps
RUN apt-get update && apt-get install -y \
    libgomp1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install the official Hugging Face CLI
RUN pip3 install huggingface_hub

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Create models directory
RUN mkdir -p /models

# --- DOWNLOAD FROM YOUR REPO ---
# Repo: prasannaJagadesh/my-model
# File: model.gguf
# Note: If your repo is PRIVATE, you must add: --token YOUR_HF_TOKEN
RUN huggingface-cli download \
    prasannaJagadesh/my-model \
    model.gguf \
    --local-dir /models \
    --local-dir-use-symlinks False

ENV LC_ALL=C.utf8
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the server pointing to your specific model file
CMD ["llama-server", "-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", "--ctx-size", "1024", "--n-gpu-layers", "0"]