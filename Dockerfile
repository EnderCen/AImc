# Use base image with Python 3.10 (for viet-tts), then create virtual envs inside
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Ho_Chi_Minh

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3.10 \
    python3-dev \
    python3-pip \
    curl \
    wget \
    && apt-get clean

# Set default python and pip to 3.10

WORKDIR /app

### ----- Setup Roop ----- ###
COPY ./roop ./roop
RUN pip install -r ./roop/requirements.txt && \
    pip uninstall -y onnxruntime onnxruntime-gpu && \
    pip install onnxruntime-gpu==1.16.0

### ----- Setup viet-tts ----- ###
COPY ./viet-tts ./viet-tts
WORKDIR /app/viet-tts
RUN pip install -e . && pip cache purge

### ----- Setup SadTalker ----- ###
COPY ./SadTalker ./SadTalker
WORKDIR /app/SadTalker
RUN pip install torch==2.0.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.0.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
    
RUN pip install -r requirements.txt

### Final Setup ###
WORKDIR /app
COPY ./main.py ./main.py
COPY ./start.sh ./start.sh
RUN chmod +x ./start.sh

CMD ["./start.sh"]
