#!/bin/bash
cd /home/claude/vanet/sim2
for st in "$@"; do
  python3 experiments4.py $st || exit 1
done
echo ALL4DONE "$@"
