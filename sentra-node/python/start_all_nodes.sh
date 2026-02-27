#!/bin/bash
# Bash script to start all nodes on Linux/macOS
# Usage: ./start_all_nodes.sh

N_NODES=5
BASE_PORT=8000

echo "======================================================================"
echo "Starting SENTRA Multi-Node Training"
echo "======================================================================"
echo "Total nodes: $N_NODES"
echo "Base port: $BASE_PORT"
echo "======================================================================"
echo ""

# Start each node in a new terminal
for i in $(seq 1 $N_NODES); do
    echo "Starting Node $i..."
    if command -v gnome-terminal &> /dev/null; then
        # GNOME Terminal
        gnome-terminal -- bash -c "cd '$PWD' && python run_node.py --node-id $i --n-nodes $N_NODES --base-port $BASE_PORT; exec bash"
    elif command -v xterm &> /dev/null; then
        # xterm
        xterm -e "cd '$PWD' && python run_node.py --node-id $i --n-nodes $N_NODES --base-port $BASE_PORT" &
    elif command -v osascript &> /dev/null; then
        # macOS Terminal
        osascript -e "tell application \"Terminal\" to do script \"cd '$PWD' && python run_node.py --node-id $i --n-nodes $N_NODES --base-port $BASE_PORT\""
    else
        # Fallback: run in background
        python run_node.py --node-id $i --n-nodes $N_NODES --base-port $BASE_PORT &
    fi
    sleep 1  # Stagger starts
done

echo ""
echo "All nodes started."
echo "To stop all nodes, press Ctrl+C or kill the processes."

