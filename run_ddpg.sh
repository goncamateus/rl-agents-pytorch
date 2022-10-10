#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "DDPG-1" -e VSS-v0
done
