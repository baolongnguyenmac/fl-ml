#!/bin/bash

# Start a Flower server
# python3 -m server.server \
#   --rounds=100 \
#   --epochs=5 \
#   --sample_fraction=0.3 \
#   --min_sample_size=1 \
#   --min_num_clients=1 \
#   --strategy='FED_AVG'  \
#   --learning_rate=0.001

python3 -m server.server \
  --rounds=100 \
  --epochs=1 \
  --sample_fraction=0.3 \
  --min_sample_size=2 \
  --min_num_clients=2 \
  --strategy='FED_META_MAML'  \
  --alpha=0.1 \
  --beta=0.01