#---------------------------------------------------------------------
# BASE IMAGE
#---------------------------------------------------------------------
ARG BASE_IMAGE=occlum/occlum:0.31.0-ubuntu22.04
ARG BASE_IMAGE_RT=occlum/occlum:0.31.0-rt-ubuntu22.04
FROM $BASE_IMAGE AS sentra-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV IS_DOCKERFILE=1

WORKDIR /

RUN apt-get update && apt-get install -y jq

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
RUN /miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN /miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /miniconda/bin/conda create --prefix /python-occlum -y  python==3.10.0 numpy==1.26.4 pyyaml==6.0.3 tensorflow==2.11.0

RUN mkdir /sentra
COPY ./sentra-node/python/node/ /sentra/
COPY ./benchmarking/ml-part/entrypoint.sh /
COPY ./benchmarking/ml-part/enclave_run_script.sh /bin/enclave_run_script.sh

WORKDIR /sentra
ENTRYPOINT ["/entrypoint.sh"]
