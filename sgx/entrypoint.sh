#!/bin/bash
/opt/occlum/start_aesm.sh
echo "starting the attestation now"
occlum run /bin/dcap_c_test
sleep infinity
echo "starting the python code now"
occlum run /bin/python3 "$@" &
