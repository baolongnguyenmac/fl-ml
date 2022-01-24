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

# alpha is only used for fedmeta meta sgd algorithm

NUM_CLIENTS=50

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client $i"
    python3 -m client.client_main \
        --cid=$i \
        --alpha=0.001 \
        --strategy_client='FedAvg' \
        --model='mnist' \
        --per_layer=1 \
        --new_client=1 &
done
echo "Started $NUM_CLIENTS clients."