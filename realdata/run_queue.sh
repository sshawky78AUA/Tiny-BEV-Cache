#!/bin/bash
# one job = one (dataset, widths) slice, each well under the 2-hour background limit
cd /home/claude/v3/dair
case "$1" in
  d1) WIDTHS=4,8,16 python3 train_tiers.py bev_intent.npz tiers_intent 30 intent40_motion.npy >> train_intent.log 2>&1 ;;
  d2) WIDTHS=32     python3 train_tiers.py bev_intent.npz tiers_intent 30 intent40_motion.npy >> train_intent.log 2>&1 ;;
  n1) WIDTHS=4,8    python3 train_tiers.py nusc_intent_fov.npz tiers_nusc 10 nusc_intent_fov_motion.npy >> train_nusc.log 2>&1 ;;
  n2) WIDTHS=16     python3 train_tiers.py nusc_intent_fov.npz tiers_nusc 10 nusc_intent_fov_motion.npy >> train_nusc.log 2>&1 ;;
  n3) WIDTHS=32     python3 train_tiers.py nusc_intent_fov.npz tiers_nusc 10 nusc_intent_fov_motion.npy >> train_nusc.log 2>&1 ;;
esac
echo "slice $1 exit $?"
