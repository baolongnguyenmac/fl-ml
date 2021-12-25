#!/bin/bash

# - strategy:
#     - FedAvg
#     - FedMetaMAML
#     - FedAvgMeta
#     - FedMetaSGD

# - model:
#     - femnist
#     - mnist
#     - cifar

python main.py \
    --num_clients=50 \
    --num_eval_clients=50 \
    --rounds=100 \
    --epochs=1 \
    --batch_size=32 \
    --fraction_fit=0 \
    --fraction_eval=0 \
    --min_fit_clients=5 \
    --min_eval_clients=50 \
    --min_available_clients=50 \
    --alpha=0.001 \
    --beta=0.001 \
    --strategy_client='FedMetaSGD' \
    --model='mnist' \
    --mode='test' \
    --per_layer=1