#!/bin/sh
peeble -c /peeble/peeble-config.json
cd /sentra/backend
python sentra-backend.py --host=0.0.0.0
