#!/bin/bash
for i in 0; do
  python train_ddpg.py --cuda -n "DDPG_Strat-1" -e VSSStrat-v0
done
