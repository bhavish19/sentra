#!/bin/bash
# Parse command-line arguments
PEBBLE_ONLY=false
args=("$@") # Save command line args

while [[ $# -gt 0 ]]; do
    case "$1" in
        -pebble_only)
            PEBBLE_ONLY=true
            shift 1
            ;;
        *)
            shift 1
            ;;
    esac
done

set -- "${args[@]}" #Restore command line args

export PEBBLE_CHAIN_LENGTH=1
export PEBBLE_VA_ALWAYS_VALID=1
export PEBBLE_WFE_NONCEREJECT=0
pebble -config /pebble/pebble-config.json  2>&1 &
PEBBLE_PID=$!
echo "Pebble is runing as process with PID $PEBBLE_PID"

if ! $PEBBLE_ONLY; then
    sleep 5
    cd /sentra/backend
    python sentra-backend.py --host=0.0.0.0 $@ &
    FLASK_PID=$!
    echo "Sentra Backend is runing as process with PID $FLASK_PID"
    trap "kill $PEBBLE_PID $FLASK_PID" 15 2

    wait $FLASK_PID
else
     wait $PEBBLE_PID
fi