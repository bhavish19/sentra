#!/bin/sh
export PEBBLE_CHAIN_LENGTH=1
export PEBBLE_VA_ALWAYS_VALID=1
export PEBBLE_WFE_NONCEREJECT=0
pebble -config /pebble/pebble-config.json  2>&1 &
PEBBLE_PID=$!
echo "Pebble is runing as process with PID $PEBBLE_PID"
cd /sentra/backend
python sentra-backend.py --host=0.0.0.0 $@ &
FLASK_PID=$!
echo "Sentra Backend is runing as process with PID $FLASK_PID"
trap "kill $PEBBLE_PID $FLASK_PID" 15 2

wait $FLASK_PID
