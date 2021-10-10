# Ubuntu 18.04 with Python 3.7
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ARG PYTHON=python3.8

# ======== #
# OS Setup #
# ======== #
RUN \
    # Make sure that we aren't prompted for timezone information
    export DEBIAN_FRONTEND=noninteractive && \
    # Install necessary tools for the build
    apt-get update && \
    apt-get install -y --no-install-recommends git-core libglib2.0-0 libsm6 libxrender-dev libxrender1 libxext6 postgresql wget && \
    apt-get install -y build-essential ${PYTHON} ${PYTHON}-dev lib${PYTHON}-dev virtualenv ${PYTHON}-venv && \
    # Clean up the apt cache
    rm -rf /var/lib/apt/lists/*

# ================== #
# Create source dirs #
# ================== #
RUN \
    mkdir /source && \
    cd /source && \
    git clone https://github.com/waikato-datamining/keras-imaging.git && \
    ${PYTHON} -m venv venv && \
    ./venv/bin/pip install --upgrade pip setuptools && \
    ./venv/bin/pip install tensorflow==2.4.0 pandas pillow scipy wai.common

ENV PYTHONPATH=/source/keras-imaging/imageclassification/src

WORKDIR /source

ENTRYPOINT ["./venv/bin/python", "./keras-imaging/imageclassification/src/sample/fine_tune_hybrid_ic.py", "/dataset.txt", "/output", "/train.txt", "/val.txt"]
