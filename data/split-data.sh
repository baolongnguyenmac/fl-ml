#!/bin/bash

python split_support_query.py --save_train shakespeare/train \
                                --leaf_train_json tmp/train \
                                --save_test shakespeare/test \
                                --leaf_test_json tmp/test \
                                --query_frac 0.2