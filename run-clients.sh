#!/bin/bash

NUM_CLIENTS=3

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python3 -m client.client \
      --cid=$i \
      --strategy='FedMetaSGD' \
      --alpha=10 \
      --model='femnist' &
done
echo "Started $NUM_CLIENTS clients."

