#!/bin/bash

# - strategy:
#     - FedAvg
#     - FedMetaMAML
#     - FedAvgMeta
#     - FedMetaSGD

# - model:
#     - femnist
#     - shakespeare
#     - sent140

python main.py \
    --num_clients=299 \
    --num_eval_clients=37 \
    --rounds=2 \
    --epochs=1 \
    --batch_size=32 \
    --fraction_fit=0 \
    --fraction_eval=0 \
    --min_fit_clients=4 \
    --min_eval_clients=4 \
    --min_available_clients=4 \
    --alpha=0.01 \
    --beta=0.001 \
    --strategy_client='FedMetaMAML' \
    --model='femnist' \
    --mode='val'