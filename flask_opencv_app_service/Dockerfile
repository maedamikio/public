FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    openssh-server \
    python3 \
    python3-pip \
    python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN echo "root:Docker!" | chpasswd && \
    mkdir /var/run/sshd && \
    sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i '$ a Ciphers aes128-cbc,3des-cbc,aes256-cbc' /etc/ssh/sshd_config && \
    sed -i '$ a MACs hmac-sha1,hmac-sha1-96' /etc/ssh/sshd_config
RUN mkdir /code
WORKDIR /code
ADD . /code/
RUN pip3 install -r requirements.txt
EXPOSE 2222 8000
CMD ["/bin/sh", "-c", "/usr/sbin/sshd && gunicorn --bind=0.0.0.0 --timeout 600 main:app"]
