#!/bin/bash
/opt/occlum/start_aesm.sh
occlum run /bin/enclave_run_script.sh "$@" #&
