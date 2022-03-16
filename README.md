# Meta-learning và Personalization layer trong Federated learning

## Thông tin khóa luận

- Tên đề tài: Meta-learning và Personalization layer trong Federated learning
- GVHD: GS. TS. Lê Hoài Bắc
- GVPB: TS. Nguyễn Tiến Huy
- Nhóm sinh viên:
    - Nguyễn Bảo Long - MSSV: 18120201
    - Cao Tất Cường - MSSV: 18120296
- Bảo vệ vào ngày 15/03/2022 tại Hội đồng Khoa học máy tính 1

## How to run

- Dữ liệu được cấu hình giống như paper: Personalized Federated Learning with Moreau Envelopes (NeurIPS 2020).

- Có 2 cách để khởi chạy simulation (đọc doc của Flower để rõ thêm):

    - Để chạy 1 mạch và khỏi suy nghĩ gì: Chạy file `run.sh`. Trong file này chứa toàn bộ lệnh để tạo ra kết quả của khoá luận. Khi chạy theo kiểu này, chương trình sẽ 

    - Trong trường hợp cần debug, có thể sử dụng 2 files `run_client.sh` và `run_server.sh`.

- Note thêm về các TH ngoại lệ của `FedPer`

## Giải thích một vài thông tin về source code

- Folder `client`: Cài đặt các thuật toán huấn luyện theo hướng Meta learning (MAML, Meta-SGD) cho client. File `client_main.py` được dùng trong mô phỏng

```text
.
├── base_client.py: 
├── client_main.py
├── fedavg_client.py
├── fedmeta_maml_client.py
└── fedmeta_sgd_client.py
```
