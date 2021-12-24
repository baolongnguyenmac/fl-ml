#!/bin/bash

# alpha is not used for fedmeta sgd

# meta learning based
python3 -m server.server_main \
    --rounds=10 \
    --epochs=1 \
    --batch_size=32 \
    --fraction_fit=0 \
    --fraction_eval=0 \
    --min_fit_clients=1 \
    --min_eval_clients=1 \
    --min_available_clients=1 \
    --alpha=0.001 \
    --beta=0.001