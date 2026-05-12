#!/bin/bash
NO_SGX=true
args=()

for arg in "$@"; do
    case "$arg" in
        -no_sgx)
            NO_SGX=true
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

set -- "${args[@]}" #Restore command line args but without -no_sgx

if ! $NO_SGX; then
    occlum run /bin/enclave_run_script.sh "$@"
else
    /python-occlum/bin/python "$@"
fi
