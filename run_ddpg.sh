#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "WithAction-Stack10-step0.01" -e VSS-v0
done
