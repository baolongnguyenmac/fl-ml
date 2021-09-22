#!/bin/bash

# Start a Flower server
python3 -m server.server \
  --rounds=100 \
  --epochs=5 \
  --sample_fraction=0.3 \
  --min_sample_size=1 \
  --min_num_clients=1 \
  --strategy='FED_AVG'  \
  --learning_rate=0.001