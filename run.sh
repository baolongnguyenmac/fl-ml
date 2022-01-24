#!/bin/bash

# - strategy: FedAvg, FedMetaMAML, FedAvgMeta, FedMetaSGD
# - model: femnist, mnist, cifar

python main.py \
    --num_clients=50 \
    --rounds=1 \
    --epochs=1 \
    --batch_size=32 \
    --min_fit_clients=5 \
    --min_eval_clients=50 \
    --min_available_clients=50 \
    --alpha=0.00001 \
    --beta=0.0005 \
    --strategy_client='FedAvgMeta' \
    --model='mnist' \
    --per_layer=1 \
    --new_client=1