#!/bin/bash

# - strategy: FedAvg, FedMetaMAML, FedAvgMeta, FedMetaSGD
# - model: femnist, mnist, cifar

# # fedMeta on local and new client (mnist)
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvg' --model='mnist' --new_client=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvg' --model='mnist' --new_client=0 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvgMeta' --model='mnist' --new_client=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvgMeta' --model='mnist' --new_client=0 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.001 --strategy_client='FedMetaMAML' --model='mnist' --new_client=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.001 --strategy_client='FedMetaMAML' --model='mnist' --new_client=0 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.0005 --strategy_client='FedMetaSGD' --model='mnist' --new_client=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.0005 --strategy_client='FedMetaSGD' --model='mnist' --new_client=0 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50

# # fedMetaPer on local and new client (mnist)
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvg' --model='mnist' --new_client=1 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50 # cant run because it cann't be fine-tuned at inference phase
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvg' --model='mnist' --new_client=0 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvgMeta' --model='mnist' --new_client=1 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.00001 --beta=0.0 --strategy_client='FedAvgMeta' --model='mnist' --new_client=0 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.001 --strategy_client='FedMetaMAML' --model='mnist' --new_client=1 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.001 --strategy_client='FedMetaMAML' --model='mnist' --new_client=0 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.0005 --strategy_client='FedMetaSGD' --model='mnist' --new_client=1 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.0005 --strategy_client='FedMetaSGD' --model='mnist' --new_client=0 --per_layer=1 --num_clients=50 --rounds=300 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50

# meta new cifar vs per new cifar
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvg' --model='cifar' --new_client=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvg' --model='cifar' --new_client=1 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50 # cant run because it cann't be fine-tuned at inference phase
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvgMeta' --model='cifar' --new_client=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvgMeta' --model='cifar' --new_client=1 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.01 --beta=0.001 --strategy_client='FedMetaMAML' --model='cifar' --new_client=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.001 --strategy_client='FedMetaMAML' --model='cifar' --new_client=1 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.01 --beta=0.001 --strategy_client='FedMetaSGD' --model='cifar' --new_client=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.01 --beta=0.001 --strategy_client='FedMetaSGD' --model='cifar' --new_client=1 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50

# per old cifar vs meta old cifar
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvg' --model='cifar' --new_client=0 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvg' --model='cifar' --new_client=0 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvgMeta' --model='cifar' --new_client=0 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.0001 --beta=0.0 --strategy_client='FedAvgMeta' --model='cifar' --new_client=0 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.005 --strategy_client='FedMetaMAML' --model='cifar' --new_client=0 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.01 --beta=0.001 --strategy_client='FedMetaMAML' --model='cifar' --new_client=0 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.01 --beta=0.01 --strategy_client='FedMetaSGD' --model='cifar' --new_client=0 --per_layer=1 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
python main.py --alpha=0.001 --beta=0.001 --strategy_client='FedMetaSGD' --model='cifar' --new_client=0 --num_clients=50 --rounds=600 --epochs=1 --batch_size=32 --min_fit_clients=5 --min_eval_clients=50 --min_available_clients=50
