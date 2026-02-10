#! /bin/bash
echo "extract attestation report"
/bin/sentra_attester
echo "run training"
/bin/python3 "$@" #&