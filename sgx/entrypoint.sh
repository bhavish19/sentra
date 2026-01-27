#!/bin/bash
/opt/occlum/start_aesm.sh
echo "extract attestation report"
occlum run /bin/sentra/attester/attester
echo "run training"
occlum run /bin/python3 "$@" #&
