#!/bin/bash
#/opt/occlum/start_aesm.sh
NO_SGX=false
args=("$@") # Save command line args

while [[ $# -gt 0 ]]; do
    case "$1" in
        -no_sgx)
            NO_SGX=true
            shift 1
            ;;
        *)
            shift 1
            ;;
    esac
done

set -- "${args[@]}" #Restore command line args

wait-for-it -t 60 sentra-backend:8888


if ! $NO_SGX; then
    occlum run /bin/enclave_run_script.sh "$@"
else
    /bin/enclave_run_script.sh "$@"
fi
