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

NUM_CLIENTS=2

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python3 -m client.client \
        --cid=$i \
        --strategy='FedMetaSGD' \
        --alpha=0.01 \
        --model='shakespeare' &
done
echo "Started $NUM_CLIENTS clients."
