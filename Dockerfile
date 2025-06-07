FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libboost-all-dev \
    libopencv-dev \
    libyaml-cpp-dev \
    ninja-build \
    python3-dev \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=/usr/local/include:/usr/local/include/eigen3:/usr/local/include/sophus:$CPLUS_INCLUDE_PATH

# Install PyTorch manually (with retries in case of timeout)
# RUN bash -c '\
#     for i in {1..5}; do \
#         pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 \
#         -f https://download.pytorch.org/whl/torch_stable.html --timeout=300 && break || sleep 10; \
#     done'

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Install Eigen from GitHub
WORKDIR /opt
RUN git clone https://gitlab.com/libeigen/eigen.git && \
    cd eigen && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install

# Install Sophus from GitHub
WORKDIR /opt
RUN git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && git checkout a621ff && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    ls /usr/local/include/sophus/se3.hpp

# Clone PoseCNN
RUN git clone https://github.com/NVlabs/PoseCNN-PyTorch.git /PoseCNN
WORKDIR /PoseCNN

# Update submodules (for ycb_render/pybind11)
RUN git submodule update --init --recursive

# Build custom layers
RUN cd lib/layers && python3 setup.py install
RUN cd lib/utils && python3 setup.py build_ext --inplace

CMD ["/bin/bash"]
