#!/bin/bash
echo "Run attester in background"
/bin/sentra_attester http://sentra-backend:8000 &
echo "run training"
/bin/python3 "$@" #&