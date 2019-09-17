FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Version
ENV CUDA=10.0.130-1 \
    CUDNN=7.6.3.30-1+cuda10.0 \
    NCCL=2.4.8-1+cuda10.0 \
    NVINFER=5.1.5-1+cuda10.0 \
    KERAS=2.2.5 \
    TENSORFLOW=1.14.0 \
    TORCH=1.2.0 \
    TORCHVISION=0.4.0 \
    OPENMPI=4.0.1 \
    HOROVOD=0.18.1

# Ubuntu, NVIDIA, Python, Open MPI, Horovod
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    g++-4.8 \
    ibverbs-providers \
    libibverbs1 \
    librdmacm1 \
    libcudnn7=$CUDNN \
    libcudnn7-dev=$CUDNN \
    libnccl-dev=$NCCL \
    libnccl2=$NCCL \
    libnvinfer-dev=$NVINFER \
    libnvinfer5=$NVINFER \
    openssh-server \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-mark hold \
    libcudnn7 \
    libcudnn7-dev \
    libnccl-dev \
    libnccl2 \
    libnvinfer-dev \
    libnvinfer5

# Machine learning
RUN pip3 install \
    keras==$KERAS \
    tensorflow-gpu==$TENSORFLOW \
    torch==$TORCH \
    torchvision==$TORCHVISION

# Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-$OPENMPI.tar.gz && \
    tar zxf openmpi-$OPENMPI.tar.gz && \
    cd openmpi-$OPENMPI && \
    ./configure && \
    make -j $(nproc) all && \
    make install && \
    rm -rf /tmp/openmpi

# Horovod
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITH_TENSORFLOW=1 \
    pip3 install horovod==$HOROVOD

# SSH
ARG AUTHORIZED_KEYS
RUN mkdir /var/run/sshd && \
    sed -i "s/#   ForwardAgent no/    ForwardAgent yes/" /etc/ssh/ssh_config && \
    sed -i "s/#   StrictHostKeyChecking ask/    StrictHostKeyChecking no/" /etc/ssh/ssh_config && \
    sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    echo "root:pass" | chpasswd && \
    mkdir /root/.ssh && \
    echo "$AUTHORIZED_KEYS" > /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    chown root:root /root/.ssh/authorized_keys

ENTRYPOINT ["/usr/sbin/sshd", "-D", "-p"]
CMD ["10022"]
