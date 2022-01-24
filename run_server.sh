#!/bin/bash

# alpha is not used for fedmeta sgd

# meta learning based
python3 -m server.server_main \
    --rounds=20 \
    --epochs=1 \
    --batch_size=32 \
    --fraction_fit=0 \
    --fraction_eval=0 \
    --min_fit_clients=5 \
    --min_eval_clients=50 \
    --min_available_clients=50 \
    --alpha=0.001 \
    --beta=0.001