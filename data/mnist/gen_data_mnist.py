import pandas as pd
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
import json

NUM_CLIENT = 50
NUM_LABEL_PER_CLIENT = 2
NUM_LABEL = 10
INTERVALS = int(NUM_CLIENT/NUM_LABEL*NUM_LABEL_PER_CLIENT)

def gen_client():
    # read data from file
    df_test: pd.DataFrame = pd.read_csv('./mnist_test.csv')
    df_train: pd.DataFrame = pd.read_csv('./mnist_train.csv')
    df = pd.concat([df_train, df_test], axis=0)

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
    write_to_file(client_dict)

    check(client_dict)

def split_num_sample_into_intervals(data_by_label: dict):
    num_sample_in_label = []

    for i in range(NUM_LABEL):
        total = len(data_by_label[i])
        tmp = []
        for j in range(INTERVALS-1):
            val = np.random.randint(total//(10 + 1), total//2)
            tmp.append(val)
            total -= val
        tmp.append(total)
        num_sample_in_label.append(tmp)

    return num_sample_in_label

# file: dict -> "users": list
#            -> "num_samples": list
#            -> "user_data": dict -> "user_id": dict -> x: (num_samples, 784)
#                                                    -> y: (num_samples,)
def write_to_file(client_dict: dict, new_client=False):
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

        for label in client_dict[user].keys():
            len_x = len(client_dict[user][label])

            if new_client:
                test_user_data[user]['x'].extend(client_dict[user][label])
                test_user_data[user]['y'].extend([label]*len_x)
            else:
                p = int(0.75*len_x)
                train_user_data[user]['x'].extend(client_dict[user][label][:p])
                train_user_data[user]['y'].extend([label]*p)

                test_user_data[user]['x'].extend(client_dict[user][label][p:])
                test_user_data[user]['y'].extend([label]*(len_x - p))

        num_train_samples.append(len(train_user_data[user]['y']))
        num_test_samples.append(len(test_user_data[user]['y']))

    all_train_data['num_samples'] = num_train_samples
    all_test_data['num_samples'] = num_test_samples
    all_train_data['user_data'] = train_user_data
    all_test_data['user_data'] = test_user_data

    if not new_client:
        with open('./train_json/train.json', 'w') as outfile:
            json.dump(all_train_data, outfile)
            outfile.close()
        with open('./test_json/test.json', 'w') as outfile:
            json.dump(all_test_data, outfile)
            outfile.close()
    else:
        with open('./test_json/new_test.json', 'w') as outfile:
            json.dump(all_test_data, outfile)
            outfile.close()

def split_data_into_classes(X, y):
    all_data = {}
    for i in set(y):
        all_data[i] = []
    for i, item in enumerate(y):
        all_data[item].append(X[i].tolist())
    return all_data

# divide all_data into clients
def split_data_into_clients(data_by_label, num_sample_in_label: list, labels = list(range(10))):
    all_user = {}
    flag1 = {labels[i]: 0 for i in labels} # trace a label
    flag2 = {labels[i]: 0 for i in labels} # trace interval in a label

    for i in range(NUM_CLIENT):
        all_user[str(i)] = {}
        idx = [i%10, i%10 + 1 if i%10 != 9 else 0]
        label_of_client = [labels[t] for t in idx]

        for label in label_of_client:
            tmp = flag1[label]
            tmp_ = tmp + num_sample_in_label[label][flag2[label]]
            all_user[str(i)][int(label)] = data_by_label[label][tmp:tmp_]
            flag1[label] = tmp_
            flag2[label] += 1
    return all_user

def check(all_client):
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(7, 7), sharey=True)
    fig.subplots_adjust(hspace=0.5)

    for i, key in enumerate(all_client.keys()):
        print(key, len(all_client[key]))
        for j in all_client[key].keys():
            print(f'{j}: {len(all_client[key][j])}', end=' ')
        print('\n')

        x = list(all_client[key].keys())
        y = [len(all_client[key][t]) for t in all_client[key].keys()]
        axes[i//10, i%10].bar(x=x, height=y)
        axes[i//10, i%10].set_title(f'{sum(y)}, {list(all_client[key].keys())}')
        axes[i//10, i%10].set_xlim(-1,10)

    plt.show()

# read file test in *.json and wrap in a np.array
def get_test_data():
    with open('./test_json/pFedMe_mnist_test.json', 'r') as input_file:
        data = json.load(input_file)
        input_file.close()

    all_data = []
    for user in data['user_data'].keys():
        feature = np.array(data['user_data'][user]['x'])
        label = np.array(data['user_data'][user]['y']).reshape(-1, 1)
        c = np.concatenate((label, feature), axis=1)
        all_data.extend(c)

    all_data = np.array(all_data)
    np.random.shuffle(all_data)
    return all_data

def gen_new_test_client():
    data = get_test_data()
    print(data.shape)

    # split into x, y
    X = data[:, 1:]
    y = data[:, 0]

    # divide data (X, y) into classes
    data_by_label = split_data_into_classes(X, y)

    # divide num_sample of each class into intervals
    num_sample_in_label = split_num_sample_into_intervals(data_by_label)

    # divide data_by_label into client
    labels = list(range(10))
    np.random.shuffle(labels)
    client_dict = split_data_into_clients(data_by_label, num_sample_in_label, labels)

    # write to file
    write_to_file(client_dict, new_client=True)

    check(client_dict)

if __name__ == '__main__':
    # gen_client()
    gen_new_test_client()
