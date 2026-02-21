#!/bin/bash

rust_args=()
python_args=()
current=1

for arg in "$@"; do
  if [ "$arg" = "---" ]; then
    current=2
    continue
  fi
  if [ "$current" -eq 1 ]; then
    rust_args+=("$arg")
  else
    python_args+=("$arg")
  fi
done

echo "Run attester in background"
/bin/sentra_attester "${rust_args[@]}" &
echo "Run training"
/bin/python3 "${python_args[@]}"