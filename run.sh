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
    --num_clients=100 \
    --num_val_clients=10 \
    --num_test_clients=10 \
    --rounds=100 \
    --epochs=1 \
    --batch_size=32 \
    --fraction_fit=0.1 \
    --fraction_eval=0.1 \
    --min_fit_clients=10 \
    --min_eval_clients=10 \
    --min_available_clients=10 \
    --alpha=0.01 \
    --beta=0.001 \
    --strategy_client='FedMetaSGD' \
    --model='femnist' \
    --mode='train'