FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

####################################################
# NO ENVIRONMENT RIGHT NOW
####################################################

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Update and install debian stuff
RUN apt-get update && apt-get -y install \
    wget \
    unzip \
    git \
    curl \
    lsb-release \
    manpages-dev \
    build-essential \
    libgl1-mesa-glx \
    mesa-utils\
    libboost-dev \
    libxerces-c-dev \
    libeigen3-dev\
    python3.10 \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install cuda dependent python stuff
COPY requirements.txt .
RUN pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html && \
    pip install torch_geometric spconv-cu118 && \
    python3 -m pip install -r requirements.txt

RUN cd /usr/include && ln -sf eigen3/Eigen Eigen