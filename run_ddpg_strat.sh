#!/bin/bash
for i in 0; do
  python train_ddpg_strat.py --cuda -n "DynAlphas-$1" -e VSSStrat-v0
done
