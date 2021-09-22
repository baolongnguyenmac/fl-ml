#!/bin/bash

NUM_CLIENTS=26

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python3 -m client.client \
      --cid=$i \
      --num_partitions=$NUM_CLIENTS \
      --strategy='FedAvg' \
      --model='shakespeare' &
done
echo "Started $NUM_CLIENTS clients."

