#!/bin/bash
#/opt/occlum/start_aesm.sh
wait-for-it sentra-backend:8888
/bin/enclave_run_script-no-sgx.sh "$@"
