#!/bin/bash
# Runs every revised sweep in order; safe to restart (each stage resumes).
cd "$(dirname "$0")"
for st in budget traces hw tref V cache dens bh; do
  python3 experiments3.py $st || exit 1
done
echo ALL3DONE
