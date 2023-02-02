# Meta-learning and Personalization layer in Federated learning

Official code for ACIIDS2022 paper "[Meta-learning and Personalization layer in Federated learning](https://link.springer.com/chapter/10.1007/978-3-031-21743-2_17)".

## General information

- Supervisor: Prof. Lê Hoài Bắc
- Reviewer: Dr. Nguyễn Tiến Huy
- Authors:
    - Nguyễn Bảo Long - MSSV: 18120201
    - Cao Tất Cường - MSSV: 18120296
- Reporting date: 15/03/2022 at Computer Science No.1, University of Science, VietNam National University - Ho Chi Minh City.

## Contact

- Corresponding author: Bao-Long Nguyen
- Email: baolongnguyen.mac@gmail.com

## How to run

- Dataset configuration: Dataset is configured as in [Personalized Federated Learning with Moreau Envelopes (NeurIPS 2020)](https://github.com/CharlieDinh/pFedMe).

- 2 ways to run simulation (read [Flower's doc](https://flower.dev/docs/) for more detail):

    - Normal mode: Run file `run.sh`. This file contains all the command codes that output the results of this thesis. After running this file, function `start_simulation()` in file `./main.py` will be called.

    - Debug mode: Run 2 files `./run_client.sh` and `./run_server.sh`. File `./run_server.sh` calls `main()` in file `./server/server_main.py` in order to create and run a server. File `./run_client.sh` creates a certain number of clients by calling function `main()` in file `./client/client_main.py` multiple times.

## Some information about source code

- Folder `./client`: Defines types of clients of FL systems (based on Flower framework).

- Folder `./client_worker`: Defines training methods (meta learning, using [learn2learn](https://github.com/learnables/learn2learn) and conventional FL training) and testing methods. These functions in here will be called by clients in `./client`.

- Folder `./data`: Contains data generator (`./data/mnist`, `./data/cifar`), and data loader (`./data/dataloaders`) for each client.

- Folder `./document`: Contains a presentation, a thesis and relevant documents.

- Folder `./experiments`: Results of `FedAvg, FedAvgMeta, FedPer, FedPerMeta, FedMeta(MAML), FedMeta(Meta-SGD), FedMeta-Per(MAML), FedMeta-Per(Meta-SGD)` running on MNIST, CIFAR-10, and on 2 types of client (new client, local client).

- Folder `./model`: Defines models and model wrapper for MNIST, CIFAR-10.

- Folder `./personalized_weight`: A temporary folder, generated during the execution of algorithms using personalization layer. This folder contains personalization layer of each client.

## Results

- We proposed `FedMeta-Per` (`FedMeta-Per(MAML), FedMeta-Per(Meta-SGD)`), a combination of Meta-learning and Personalization layer into a FL system.

### MNIST

- Classification results (%) of local client using MNIST dataset

|                            | $acc_{micro}$       | $acc_{macro}$             | $P_{macro}$               | $R_{macro}$               | $F1_{macro}$               |
| :------------------------- | :-----------------: | :-----------------------: | :-----------------------: | :-----------------------: | :------------------------: |
| FedAvg                     | 85.03               | 82.14±14.76               | 82.03±13.88               | 81.54±14.33               | 79.43±16.83                |
| FedPer                     | 77.29               | 75.48±14.84               | 76.07±14.99               | 74.01±15.13               | 72.32±15.99                |
| FedAvgMeta                 | 84.84               | 81.56±16.68               | 80.71±17.02               | 81.18±16.16               | 78.31±19.8                 |
| FedPerMeta                 | 75.91               | 74.11±16.2                | 75.68±15.94               | 72.93±15.58               | 71.22±16.77                |
| FedMeta(MAML)              | 92.99               | 91.14±5.99                | 90.56±6.24                | 90.98±5.9                 | 90.16±6.28                 |
| FedMeta(Meta-SGD)          | 98.02               | 96.35±4.62                | 96.49±4.1                 | 95.64±5.94                | 95.80±5.51                 |
| **FedMeta-Per(MAML)**      | **99.37**           | **99.12±1.29**            | **99.11±1.3**             | **98.82±1.99**            | **98.94±1.6**              |
| **FedMeta-Per(Meta-SGD)**  | 98.92               | 98.15±3.32                | 98.42±1.95                | 98.42±1.96                | 98.20±2.94                 |

- Classification results (%) on new client using MNIST dataset

|                            | $acc_{micro}$       | $acc_{macro}$        | $P_{macro}$             | $R_{macro}$             | $F1_{macro}$             |
| :------------------------- | :--------------: | :---------------------: | :---------------------: | :---------------------: | :----------------------: |
| FedAvg                     | 83\.92           | 81\.69±19.71            | 79\.57±20.18            | 80\.46±17.84            | 77\.66±22.54             |
| FedPer                     | 78\.3            | 76\.19±18.79            | 75\.91±17.52            | 74\.73±17.32            | 72\.72±19.3              |
| FedAvgMeta                 | 84\.34           | 82\.37±17.42            | 81\.38±16.25            | 80\.91±15.62            | 78\.78±19.31             |
| FedPerMeta                 | 77\.47           | 75\.56±20.33            | 75\.09±19.52            | 74\.92±18.85            | 72\.60±21.37             |
| FedMeta(MAML)              | 92\.96           | 91\.88±5.88             | 90\.14±7.97             | 90\.74±5.95             | 90\.02±7.34              |
| FedMeta(Meta-SGD)          | 96\.39           | 93\.53±8.39             | 93\.73±10.26            | 88\.65±14.06            | 89\.31±14.56             |
| **FedMeta-Per(MAML)**      | 93\.6            | 93\.57±5.58             | 93\.64±5.56             | 90\.98±6.98             | 91\.83±6.43              |
| **FedMeta-Per(Meta-SGD)**  | **96\.62**       | **95\.88±3.58**         | **95\.73±4.11**         | **94\.34±5.05**         | **94\.85±4.61**          |

### CIFAR-10

- Classification results (%) of local client using CIFAR-10 dataset

|                            | $acc_{micro}$    | $acc_{macro}$            | $P_{macro}$              | $R_{macro}$              | $F1_{macro}$         |
| :------------------------- | :--------------: | :----------------------: | :----------------------: | :----------------------: | :------------------: |
| FedAvg                     | 19\.02           | 19\.29±25.11             | 15\.57±23.7              | 20\.65±25.55             | 16\.85±23.92         |
| FedPer                     | 13\.22           | 12\.99±19.39             | 18\.34±28.59             | 14\.14±20.83             | 10\.52±14.91         |
| FedAvgMeta                 | 40\.3            | 38\.47±31.52             | 32\.84±32.06             | 39\.33±30.35             | 33\.81±30.61         |
| FedPerMeta                 | 18\.57           | 17\.48±22.55             | 20\.02±27.4              | 18\.43±23.47             | 14\.54±18.67         |
| FedMeta(MAML)              | 69\.02           | 68\.76±14.86             | 67\.42±21.16             | 66\.56±13.48             | 61\.14±20            |
| FedMeta(Meta-SGD)          | 78\.63           | 78\.73±11.59             | 74\.65±21.12             | 75\.25±14.09             | 72\.87±18.31         |
| **FedMeta-Per(MAML)**      | **86\.6**        | **86\.52±6.31**          | **86\.43±5.88**          | **85\.47±6.87**          | **85\.33±6.77**      |
| **FedMeta-Per(Meta-SGD)**  | 85\.61           | 85\.68±7.22              | 86\.26±6.35              | 85\.36±6.83              | 85\.08±7.32          |

- Classification results (%) of new client using CIFAR-10 dataset

|                            | $acc_{micro}$    | $acc_{macro}$            | $P_{macro}$              | $R_{macro}$              | $F1_{macro}$         |
| :------------------------- | :---------------: | :----------------------: | :----------------------: | :----------------------: | :-----------------------: |
| FedAvg                     | 24\.63            | 24\.83±22.57             | 18\.36±20.15             | 24\.44±21.95             | 20\.52±20.45              |
| FedPer                     | 14\.4             | 14\.52±20.15             | 12\.59±20.65             | 14\.23±19.58             | 10\.66±13.79              |
| FedAvgMeta                 | 43\.39            | 43\.54±18                | 33\.45±21.44             | 42\.87±16.98             | 35\.14±17.22              |
| FedPerMeta                 | 13\.33            | 13\.57±19.62             | 11\.99±19.52             | 13\.53±19.08             | 10\.05±13.17              |
| FedMeta(MAML)              | 61\.69            | 61\.64±12.49             | 52\.66±26.06             | 59\.94±12.35             | 50\.76±19.2               |
| FedMeta(Meta-SGD)          | 68\.36            | 67\.89±15.11             | **70\.3±22.37**          | 66\.86±15.02             | 60\.24±21.52              |
| **FedMeta-Per(MAML)**      | 64\.22            | 63\.70±12.29             | 57\.06±24.99             | 61\.63±12.66             | 53\.68±19.06              |
| **FedMeta-Per(Meta-SGD)**  | **69\.97**        | **69\.13±14.63**         | 66\.53±24.91             | **67\.82±15.34**         | **62\.42±20.94**          |

### Visualization

- `FedMeta-Per` vs. (`FedAvg`, `FedAvgMeta`, `FedPer`, `FedPerMeta`): The proposed methods achieves higher degree in term of convergence and accuracy compared with `FedAvg` and `FedPer`.

![](./document/thesis/images/sum1.png)

- `FedMeta-Per` vs. `FedMeta`: Improved personalization is the reason why results on local clients of `FedMeta-Per` achieve faster convergence and higher accuracy than `FedMeta`. Regarding the new clients, 2 algorithms achieve the same degree of convergence. However, the personalization layer at each `FedMeta-Per` client will improve over time as the client participates in one or more local training step (new client becomes local client).

![](./document/thesis/images/sum2.png)
