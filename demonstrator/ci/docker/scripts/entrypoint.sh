#!/bin/sh
export PEBBLE_CHAIN_LENGTH=1
export PEBBLE_VA_ALWAYS_VALID=1
pebble -config /pebble/pebble-config.json  2>&1 &
PEBBLE_PID=$!

cd /sentra/backend
python sentra-backend.py --host=0.0.0.0
kill $PEBBLE_PID
