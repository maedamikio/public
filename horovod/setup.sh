#!/bin/sh

# Version
CUDA=10.0.130-1
CUDNN=7.6.3.30-1+cuda10.0
NCCL=2.4.8-1+cuda10.0
NVINFER=5.1.5-1+cuda10.0
KERAS=2.2.5
TENSORFLOW=1.14.0
TORCH=1.2.0
TORCHVISION=0.4.0
OPENMPI=4.0.1
HOROVOD=0.18.1

# Ubuntu
sudo apt update
sudo apt upgrade -y
sudo apt dist-upgrade -y
sudo apt install -y --no-install-recommends ca-certificates curl openssh-server wget
sudo apt autoremove -y

# Docker
sudo apt remove -y docker docker-engine docker.io containerd runc
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER

# NVIDIA
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-mark hold cuda-repo-ubuntu1804 nvidia-machine-learning-repo-ubuntu1804
sudo apt update
sudo apt install -y --no-install-recommends nvidia-driver-430
sudo apt install -y --no-install-recommends cuda=$CUDA libcudnn7=$CUDNN libcudnn7-dev=$CUDNN libnccl-dev=$NCCL libnccl2=$NCCL libnvinfer-dev=$NVINFER libnvinfer5=$NVINFER
sudo apt-mark hold cuda libcudnn7 libcudnn7-dev libnccl-dev libnccl2 libnvinfer-dev libnvinfer5
echo "#!/bin/sh -e\nnvidia-smi -pm 1\nexit 0" | sudo tee /etc/init.d/nvidia && sudo chmod 755 /etc/init.d/nvidia && sudo ln -s ../init.d/nvidia /etc/rc5.d/S01nvidia

# NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Python
sudo apt install -y --no-install-recommends python3 python3-dev python3-pip python3-setuptools

# Machine learning
sudo pip3 install keras==$KERAS tensorflow-gpu==$TENSORFLOW torch==$TORCH torchvision==$TORCHVISION

# Open MPI
sudo apt install -y --no-install-recommends build-essential
wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-$OPENMPI.tar.gz
tar zxf openmpi-$OPENMPI.tar.gz
cd openmpi-$OPENMPI
./configure
make -j $(nproc) all
sudo make install

# Horovod
sudo apt install -y --no-install-recommends build-essential g++-4.8 ibverbs-providers libibverbs1 librdmacm1
sudo HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 pip3 install horovod==$HOROVOD

# SSH
sudo sed -i "s/#   ForwardAgent no/    ForwardAgent yes/" /etc/ssh/ssh_config
sudo sed -i "s/#   StrictHostKeyChecking ask/    StrictHostKeyChecking no/" /etc/ssh/ssh_config

# MNIST
mkdir $HOME/examples
cd $HOME/examples
wget https://raw.githubusercontent.com/keras-team/keras/master/examples/mnist_cnn.py -O mnist_keras.py
wget https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py -O mnist_pytorch.py
wget https://raw.githubusercontent.com/horovod/horovod/master/examples/keras_mnist.py -O mnist_keras_horovod.py
wget https://raw.githubusercontent.com/horovod/horovod/master/examples/pytorch_mnist.py -O mnist_pytorch_horovod.py
