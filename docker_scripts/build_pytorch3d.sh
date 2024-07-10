#!/bin/bash

set -xueo pipefail

BUILD_DIR=~/pytorch3d_build

export PYTHONDONTWRITEBYTECODE=1

mkdir -p $BUILD_DIR
(
    cd $BUILD_DIR

    git clone --depth=1 -b v0.7.3 https://github.com/facebookresearch/pytorch3d.git .
    
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="Ampere;Turing;Volta;Pascal"
    export CUDA_HOME="/usr/local/cuda"
    pip install .
)
rm -rf $BUILD_DIR
