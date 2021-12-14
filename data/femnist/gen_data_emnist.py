import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

NUM_CLIENT = 93
NUM_LABEL_PER_CLIENT = 10
NUM_LABEL = 62
INTERVALS = int(NUM_CLIENT/NUM_LABEL*NUM_LABEL_PER_CLIENT)

def main():
    np.random.seed(10)

    # read data from file
    df = pd.read_csv('./emnist-byclass-test.csv')

    # convert to numpy array
    raw_data = df.to_numpy(copy=False)

    # normalization
    X = raw_data[:, 1:]/255.0
    y = raw_data[:, 0]

    # divide data (X, y) into classes
    data_by_label = split_data_into_classes(X, y)

    # divide num_sample of each class into intervals
    num_sample_in_label = split_num_sample_into_intervals(data_by_label)

    # divide data_by_label into client
    client_dict = split_data_into_clients(data_by_label, num_sample_in_label)

    # write to file
    # write_to_file(client_dict)

    check(client_dict)

def split_num_sample_into_intervals(data_by_label: dict):
    num_sample_in_label = []

    for i in range(NUM_LABEL):
        total = len(data_by_label[i])
        tmp = []
        for j in range(INTERVALS-1):
            # val = np.random.randint(total//(50 + 1), total//12)
            val = np.random.randint(total//(INTERVALS+5), total//(INTERVALS-9))
            tmp.append(val)
            total -= val
        tmp.append(total)
        num_sample_in_label.append(tmp)

    return num_sample_in_label

def split_data_into_classes(X, y):
    all_data = {}
    for i in set(y):
        all_data[i] = []
    for i, item in enumerate(y):
        all_data[item].append(X[i].tolist())
    return all_data

def split_data_into_clients(data_by_label, num_sample_in_label: list):
    # divide all_data into clients
    all_client = {}
    flag1 = [0]*NUM_LABEL  # flag for data per label
    flag2 = [0]*NUM_LABEL  # flag for num_sample_in_label

    for i in range(NUM_CLIENT):
        all_client[str(i)] = {}

        l = i%NUM_LABEL
        for j in range(NUM_LABEL_PER_CLIENT):
            tmp = flag1[l]
            tmp_ = tmp + num_sample_in_label[l][flag2[l]]

            all_client[str(i)][l] = data_by_label[l][tmp:tmp_]

            flag2[l] += 1
            flag1[l] = tmp_
            l = l+1 if l < NUM_LABEL-1 else 0

    return all_client

# file: dict -> "users": list
#            -> "num_samples": list
#            -> "user_data": dict -> "user_id": dict -> x: (num_samples, 784)
#                                                    -> y: (num_samples,)
def write_to_file(client_dict: dict):
    all_train_data = {}
    all_test_data = {}

    users = list(client_dict.keys())
    all_train_data['users'] = users
    all_test_data['users'] = users

    num_train_samples = []
    num_test_samples = []
    train_user_data = {}
    test_user_data = {}
    for user in users:
        train_user_data[user] = {}
        test_user_data[user] = {}

        train_user_data[user]['x'] = []
        train_user_data[user]['y'] = []
        test_user_data[user]['x'] = []
        test_user_data[user]['y'] = []

        for key in client_dict[user].keys():
            len_x = len(client_dict[user][key])
            p = int(0.8*len_x)

            train_user_data[user]['x'].extend(client_dict[user][key][:p])
            train_user_data[user]['y'].extend([key]*p)

            test_user_data[user]['x'].extend(client_dict[user][key][p:])
            test_user_data[user]['y'].extend([key]*(len_x - p))

        num_train_samples.append(len(train_user_data[user]['y']))
        num_test_samples.append(len(test_user_data[user]['y']))

    all_train_data['num_samples'] = num_train_samples
    all_test_data['num_samples'] = num_test_samples
    all_train_data['user_data'] = train_user_data
    all_test_data['user_data'] = test_user_data

    with open('./train_json/train.json', 'w') as outfile:
        json.dump(all_train_data, outfile)
    with open('./test_json/test.json', 'w') as outfile:
        json.dump(all_test_data, outfile)

def check(all_client):
    fig, axes = plt.subplots(nrows=11, ncols=30, figsize=(15, 15), sharey=True)
    fig.subplots_adjust(hspace=0.2)

    for i, key in enumerate(all_client.keys()):
        print(key, len(all_client[key]))
        for j in all_client[key].keys():
            print(f'{j}: {len(all_client[key][j])}', end=' ')
        print('\n')

        x = list(range(NUM_LABEL_PER_CLIENT))
        y = [len(all_client[key][t]) for t in all_client[key].keys()]
        axes[int(i/30), int(i % 30)].bar(x=x, height=y)

    plt.show()

if __name__ == '__main__':
    main()
