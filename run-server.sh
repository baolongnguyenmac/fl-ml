#!/bin/bash

# - strategy:
#     - FED_AVG
#     - FED_META_MAML
#     - FED_META_SGD

# # fedAvg stuff
# python3 -m server.server \
#     --rounds=3 \
#     --epochs=2 \
#     --sample_fraction=0.3 \
#     --min_sample_size=2 \
#     --min_num_clients=2 \
#     --strategy='FED_AVG' \
#     --learning_rate=0.001

# meta learning based
python3 -m server.server \
    --rounds=5 \
    --epochs=2 \
    --sample_fraction=0.3 \
    --min_sample_size=2 \
    --min_num_clients=2 \
    --strategy='FED_META_SGD' \
    --alpha=0.01 \
    --beta=10