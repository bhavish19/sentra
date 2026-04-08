#!/bin/bash
#/opt/occlum/start_aesm.sh
wait-for-it sentra-backend:8888
occlum run /bin/enclave_run_script.sh "$@"
