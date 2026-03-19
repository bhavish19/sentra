#---------------------------------------------------------------------
# BASE IMAGE
#---------------------------------------------------------------------
ARG BASE_IMAGE=ubuntu:22.04
FROM $BASE_IMAGE AS sentra-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV IS_DOCKERFILE=1

WORKDIR /

RUN apt-get update

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
RUN /miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN /miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /miniconda/bin/conda create --prefix /python-occlum -y  python=3.10.0 numpy==1.26.4 pyyaml==6.0.3 pip

COPY ./sentra-node/docker/sentra-node-sbom.yaml /
RUN mkdir /sentra
RUN mkdir /sentra/ml_training
COPY ./sentra-node/python/ml_training /sentra/ml_training/
COPY ./sentra-node/python/pyproject.toml /sentra/
COPY ./sentra-node/docker/node_config.yaml /sentra/
COPY ./sentra-node/docker/training_config.yaml /sentra/
COPY ./sentra-node/python/run_mnist_batched_secure.py /sentra/
COPY ./demonstrator/backend/sentrabackend/client_config.yaml /sentra/

COPY ./sentra-node/docker/training_config.yaml /etc/training_config.yaml
RUN chmod 644 /etc/training_config.yaml

#Build attester
#Just update crates.io (and cache it...)
RUN mkdir -p /sentra-node/rust/src
COPY ./sentra-node/rust/Cargo.toml /sentra-node/rust
COPY ./sentra-node/rust/rust-toolchain.toml /sentra-node/rust
COPY ./sentra-node/rust/build.rs /sentra-node/rust
COPY ./sentra-node/rust/SentraInterNode-GRPC-Services.proto /sentra-node/rust/
COPY ./demonstrator/backend/SentraBackend-GRPC-Services.proto /sentra-node/rust/

WORKDIR /sentra-node/rust
RUN echo 'fn main() {}' > ./src/main.rs
#RUN cargo update -p socket2@0.6.2 --precise 0.5.10
RUN cargo update -p getrandom@0.4.2 --precise 0.3.4
#RUN occlum-cargo update -p prost-types@0.13.5 --precise 0.12.3
RUN cargo build --release

COPY ./sentra-node/rust /sentra-node/rust
#RUN rm rust-toolchain.toml
RUN rm SentraBackend-GRPC-Services.proto
COPY ./demonstrator/backend/SentraBackend-GRPC-Services.proto /sentra-node/rust/

RUN cargo build --release
COPY ./demonstrator/ci/docker/config/pebble/pebble.cer /sentra-node/rust/
COPY ./sentra-node/docker/enclave_run_script.sh /

WORKDIR /sentra
RUN /python-occlum/bin/pip install .

#RUN occlum new /occlum-instance
#RUN rm -rf /occlum-instance/image
#WORKDIR /occlum-instance

#RUN mkdir -p ./image

#RUN copy_bom -f /sentra-node-sbom.yaml --root image --include-dir /opt/occlum/etc/template

#RUN new_json="$(jq '.metadata.debuggable=false \
#    |.feature.enable_edmm=true \
#    |.resource_limits.user_space_size = "1MB" \
#    |.resource_limits.user_space_max_size = "5400MB" \
#    |.resource_limits.kernel_space_heap_size = "1MB" \
#    |.resource_limits.kernel_space_heap_max_size = "512MB" \
#    |.resource_limits.max_num_of_threads = 64 \
#    |.env.default += ["PYTHONHOME=/opt/python-occlum", "OMP_NUM_THREADS=1",  "PATH=/opt/python-occlum/bin:/bin:/usr/bin"]' Occlum.json)" \
#    && echo "${new_json}" > Occlum.json

#RUN ENABLE_EDMM=Y occlum build
#RUN occlum package --debug occlum-instance.tar.gz

#FROM mikefarah/yq:latest AS yq-source

#---------------------------------------------------------------------
# TARGET IMAGE
#---------------------------------------------------------------------
FROM $BASE_IMAGE AS sentra
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

#COPY --from=yq-source /usr/bin/yq /usr/bin
RUN apt-get update && apt-get install -y wait-for-it

#RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu jammy main' | tee /etc/apt/sources.list.d/intel-sgx.list

#RUN apt-get update && apt-get install -y \
#sgx-aesm-service=2.21.100.1-jammy1 wait-for-it


#COPY --from=sentra-builder /opt/occlum/start_aesm.sh /opt/occlum/
COPY ./sentra-node/docker/entrypoint-no-sgx.sh /
#COPY ./sentra-node/docker/sgx_sentra_qcnl.conf /etc/sgx_default_qcnl.conf
#RUN mkdir -p /var/run/aesmd

#COPY --from=sentra-builder /occlum-instance/occlum-instance.tar.gz /

WORKDIR /
#RUN tar -xf occlum-instance.tar.gz
#RUN rm occlum-instance.tar.gz

#WORKDIR /occlum-instance
ENTRYPOINT ["/entrypoint-no-sgx.sh"]
