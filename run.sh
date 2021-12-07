#!/bin/bash

# - strategy:
#     - FedAvg
#     - FedMetaMAML
#     - FedAvgMeta
#     - FedMetaSGD

# - model:
#     - femnist
#     - mnist
#     - sent140

python main.py \
    --num_clients=20 \
    --num_eval_clients=20 \
    --rounds=100 \
    --epochs=1 \
    --batch_size=32 \
    --fraction_fit=0 \
    --fraction_eval=0 \
    --min_fit_clients=4 \
    --min_eval_clients=4 \
    --min_available_clients=4 \
    --alpha=0.00001 \
    --beta=0.001 \
    --strategy_client='FedAvg' \
    --model='mnist' \
    --mode='val'