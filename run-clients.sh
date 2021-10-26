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

NUM_CLIENTS=2

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client $i"
    python3 -m client.client_main \
        --cid=$i \
        --alpha=0.01 \
        --strategy='FedMetaSGD' \
        --model='femnist' &
done
echo "Started $NUM_CLIENTS clients."
