#!/bin/sh
docker run --rm -it --device /dev/sgx_enclave:/dev/sgx_enclave --device /dev/sgx_provision:/dev/sgx_provision sentra-ml-benchmark:latest $@