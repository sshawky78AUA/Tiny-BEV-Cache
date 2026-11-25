#!/bin/bash
cd "$(dirname "$0")"
for st in "$@"; do
  python3 experiments4.py $st || exit 1
done
echo ALL4DONE "$@"
