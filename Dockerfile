
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    cmake \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip
RUN pip3 install torch==2.0.0+cu117 torchvision --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install ninja
COPY docker_scripts/build_pytorch3d.sh /tmp/ds/
RUN bash /tmp/ds/build_pytorch3d.sh

RUN pip3 install --upgrade "pip==24.0.*"
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY mesh_extraction /home/mv2mp/mesh_extraction
RUN python3 /home/mv2mp/mesh_extraction/setup.py install

COPY docker_scripts /tmp/ds/
RUN bash /tmp/ds/patch_pytorch_lightning.sh

WORKDIR /home/mv2mp
COPY . /home/mv2mp/

ENTRYPOINT ["python3", "train.py"]
