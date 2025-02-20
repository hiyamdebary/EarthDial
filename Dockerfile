# Use the latest stable PyTorch image with CUDA 11.8 and cuDNN 8
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# Set the working directory
WORKDIR /earthdial

# Update system packages and install basic utilities
RUN apt-get update && apt-get install -y \
    git \
    nano \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for caching
ENV TMPDIR=/cos/Model_Files/Model_Weights/cache
ENV HUGGING_FACE_HUB_TOKEN=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
ENV HF_DATASETS_CACHE=/cos/Model_Files/Model_Weights/cache
ENV HF_MODULES_CACHE=/cos/Model_Files/Model_Weights/cache
ENV HF_HUB_CACHE=/cos/Model_Files/Model_Weights/cache
ENV HF_HOME=/cos/Model_Files/Model_Weights/cache
ENV XDG_CACHE_HOME=/cos/Model_Files/Model_Weights/cache
ENV TRANSFORMERS_CACHE=/cos/Model_Files/Model_Weights/cache
ENV PYTORCH_TRANSFORMERS_CACHE=/cos/Model_Files/Model_Weights/cache
ENV DOCKER_BUILDKIT=0

# Upgrade pip
RUN pip install --upgrade pip 

# Install OpenCV and other Python dependencies
RUN pip install opencv-contrib-python

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install FlashAttention library
RUN pip install flash-attn==2.3.6 --no-build-isolation

# Default command
CMD ["bash"]
