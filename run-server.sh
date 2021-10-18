#!/bin/bash

# alpha is not used for fedmeta sgd

# meta learning based
python3 -m server.server_main \
    --rounds=10 \
    --epochs=2 \
    --batch_size=32 \
    --sample_fraction=0.3 \
    --min_sample_size=2 \
    --min_num_clients=2 \
    --alpha=0.01 \
    --beta=0.001
