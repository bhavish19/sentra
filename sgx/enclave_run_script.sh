#!/bin/bash
echo "Run attester in background"
/bin/sentra_attester &
echo "run training"
/bin/python3 "$@" #&