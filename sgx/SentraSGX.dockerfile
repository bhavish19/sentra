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
RUN /miniconda/bin/conda create --prefix /python-occlum -y  python=3.10.0 numpy==1.26.4 pyyaml==6.0.3

COPY ./sgx/sentra-sbom.yaml /
RUN mkdir /sentra
RUN mkdir /sentra/ml_training
COPY ./ml_training /sentra/ml_training/
COPY ./run_training.py /sentra/
COPY ./run_dp_training.py /sentra/
COPY ./run_node.py /sentra/
COPY ./sgx/node_config.yaml /sentra/

COPY ./attestation/attester /attester
WORKDIR /attester
RUN cargo build --release

RUN occlum new /occlum-instance
RUN rm -rf /occlum-instance/image
WORKDIR /occlum-instance

RUN mkdir -p image/bin
RUN cp /bin/ls image/bin/
RUN copy_bom -f /sentra-sbom.yaml --root image --include-dir /opt/occlum/etc/template 

RUN new_json="$(jq '.resource_limits.user_space_size = "1MB" |.resource_limits.user_space_max_size = "5400MB" |.resource_limits.kernel_space_heap_size = "1MB" |.resource_limits.kernel_space_heap_max_size = "512MB" |.resource_limits.max_num_of_threads = 64 |.env.default += ["PYTHONHOME=/opt/python-occlum", "OMP_NUM_THREADS=1"]' Occlum.json)" && echo "${new_json}" > Occlum.json

RUN ENABLE_EDMM=Y occlum build
RUN occlum package --debug occlum-instance.tar.gz

FROM mikefarah/yq:latest AS yq-source

#---------------------------------------------------------------------
# TARGET IMAGE
#---------------------------------------------------------------------
FROM $BASE_IMAGE_RT AS sentra
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

COPY --from=yq-source /usr/bin/yq /usr/bin

COPY --from=sentra-builder /occlum-instance/occlum-instance.tar.gz /

WORKDIR /
RUN tar -xf occlum-instance.tar.gz
RUN rm occlum-instance.tar.gz

RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu jammy main' | tee /etc/apt/sources.list.d/intel-sgx.list

RUN apt-get update && apt-get install -y \
sgx-aesm-service=2.21.100.1-jammy1


COPY --from=sentra-builder /opt/occlum/start_aesm.sh /opt/occlum/
COPY ./sgx/entrypoint.sh /
COPY ./sgx/sgx_sentra_qcnl.conf /etc/sgx_default_qcnl.conf
RUN mkdir -p /var/run/aesmd

WORKDIR /occlum-instance    
ENTRYPOINT ["/entrypoint.sh"]
