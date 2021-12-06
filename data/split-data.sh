#!/bin/bash

python split_support_query.py --save_train mnist/train \
                                --leaf_train_json data/train \
                                --save_test mnsit/test \
                                --leaf_test_json data/test \
                                --query_frac 0.2