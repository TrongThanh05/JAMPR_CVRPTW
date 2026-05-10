# JAMPR for CVRPTW

> **Joint Attention Model for Parallel Route-Construction** — giải bài toán **Vehicle Routing Problem with Time Windows (VRPTW)** bằng Deep Reinforcement Learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Tổng quan

Dự án này triển khai lại kiến trúc **JAMPR** (Joint Attention Model for Parallel Route-Construction) để giải các biến thể của bài toán **CVRPTW** (Capacitated VRP with Time Windows).

Model sử dụng **Policy Gradient (REINFORCE)** với **Rollout Baseline** để học cách xây dựng đồng thời nhiều tour xe từ đầu, tối ưu hóa tổng chi phí di chuyển và các vi phạm ràng buộc thời gian.

### Các bài toán được hỗ trợ

| Tên | Mô tả | Penalty α | Penalty β |
|-----|-------|-----------|-----------|
| `cvrp` | Capacitated VRP (không có TW) | — | — |
| `cvrptw_tw1` | Hard Time Windows — xe chỉ phục vụ trong `[a_i, b_i]`, chờ nếu sớm, bỏ nếu muộn | 1.0 | ∞ |
| `cvrptw_tw2` | Soft Upper Bound — phạt tuyến tính nếu đến muộn hơn `b_i` | 0.0 | 0.5 |
| `cvrptw_tw3` | Soft Full — phạt cả hai chiều sớm và muộn | 0.1 | 0.5 |

---

## 🏗️ Kiến trúc

```
JAMPR Model
├── Node Encoder      (Static)   — mã hóa thông tin khách hàng qua Self-Attention
├── Tour Encoder      (Dynamic)  — mã hóa lịch sử tour hiện tại của từng xe
├── Vehicle Encoder   (Dynamic)  — mã hóa trạng thái từng xe (vị trí, thời gian, tải)
├── Context Builder              — tổng hợp graph-level + fleet-level embedding
├── Affinity Module              — tính điểm cho từng cặp (xe, khách hàng)
└── Decoder (Attention)          — chọn hành động tiếp theo qua Masked Softmax
```

**Joint Action Space:** Tại mỗi bước, model xét đồng thời `m_con` xe đang hoạt động × N khách hàng, chọn cặp `(xe k*, khách hàng n*)` tối ưu nhất.

---

## 📁 Cấu trúc dự án

```
JAMPR_FOR_VRPTW/
├── configs/
│   ├── data_config.yaml        # Cấu hình dữ liệu (kích thước, seed, tham số TW)
│   ├── model_config.yaml       # Cấu hình model (embedding dim, heads, m_con, ...)
│   └── training_config.yaml    # Cấu hình training (lr, epochs, batch size, ...)
│
├── scripts/
│   ├── train.py                # Script huấn luyện chính
│   ├── evaluate.py             # Script đánh giá và benchmark
│   ├── generate_data.py        # Script tạo dữ liệu val/test
│   └── visualize.py            # Trực quan hóa tuyến đường trên bản đồ
│
├── src/
│   ├── data/
│   │   ├── generator.py        # Sinh dữ liệu ngẫu nhiên cho CVRP & CVRPTW
│   │   └── dataset.py          # PyTorch Dataset wrapper
│   ├── models/
│   │   ├── jampr.py            # Model JAMPR chính (forward, decode, step)
│   │   ├── node_encoder.py     # Node Encoder (3-layer Self-Attention)
│   │   ├── vehicle_encoder.py  # Vehicle Encoder (Feed-Forward)
│   │   ├── attention.py        # Multi-Head Attention module
│   │   ├── decoder.py          # Attention Decoder với masking
│   │   └── affinity.py         # Affinity function g_a(vehicle, node)
│   ├── training/
│   │   ├── trainer.py          # Training loop chính
│   │   ├── baseline.py         # Rollout Baseline (Kool et al. 2019)
│   │   ├── reinforce.py        # REINFORCE loss computation
│   │   └── scheduler.py        # Learning rate scheduler
│   ├── evaluation/
│   │   └── evaluator.py        # Greedy & Sampling evaluation
│   └── utils/
│       ├── seed.py             # Reproducibility seed
│       └── logging_utils.py    # TensorBoard + console logging
│
├── outputs/                    # (auto-generated) checkpoints, logs, data
│   ├── checkpoints/            # Model checkpoints (.pt)
│   ├── logs/                   # TensorBoard logs
│   └── data/                   # Generated datasets (.pt)
│
├── tests/                      # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Yêu cầu hệ thống

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0.0
- CPU (đủ để chạy debug/nhỏ) hoặc GPU với CUDA (khuyến nghị cho training đầy đủ)

---

## 🚀 Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/TrongThanh05/JAMPR_CVRPTW.git
cd JAMPR_CVRPTW
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài dưới dạng package:

```bash
pip install -e .
```

> **Lưu ý PyTorch với CUDA:** Nếu máy có GPU, cài PyTorch theo hướng dẫn tại [pytorch.org](https://pytorch.org/get-started/locally/) để dùng đúng phiên bản CUDA.

---

## 🏃 Cách chạy

### Bước 1: Tạo dữ liệu validation/test (tùy chọn)

```bash
# Tạo 10,000 instances validation cho CVRPTW TW1, N=20 khách hàng
python scripts/generate_data.py --problem cvrptw_tw1 --n 20 --split val --size 10000

# Tạo test set
python scripts/generate_data.py --problem cvrptw_tw1 --n 20 --split test --size 10000
```

> Nếu bỏ qua bước này, training vẫn chạy được — validation set sẽ được sinh tự động.

---

### Bước 2: Training

**Chạy nhanh (debug mode — vài phút, CPU):**

```bash
python scripts/train.py --problem cvrptw_tw1 --n 20 --debug
```

**Chạy đầy đủ (CPU-light, ~30–60 phút):**

```bash
python scripts/train.py --problem cvrptw_tw1 --n 20
```

**Các tham số:**

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--problem` | Loại bài toán: `cvrp`, `cvrptw_tw1`, `cvrptw_tw2`, `cvrptw_tw3` | `cvrptw_tw1` |
| `--n` | Số khách hàng: `20` hoặc `50` | `20` |
| `--debug` | Bật debug mode (siêu nhỏ, chạy vài giây) | `False` |
| `--resume` | Đường dẫn tới checkpoint để tiếp tục training | `None` |
| `--config` | Đường dẫn config file | `configs/training_config.yaml` |

**Tiếp tục training từ checkpoint:**

```bash
python scripts/train.py --problem cvrptw_tw1 --n 20 --resume outputs/checkpoints/best.pt
```

**Giám sát với TensorBoard:**

```bash
tensorboard --logdir outputs/logs
```

---

### Bước 3: Đánh giá

```bash
# Greedy evaluation
python scripts/evaluate.py --model outputs/checkpoints/best.pt --problem cvrptw_tw1 --n 20

# Sampling evaluation (1280 samples, chất lượng cao hơn, chậm hơn)
python scripts/evaluate.py --model outputs/checkpoints/best.pt --problem cvrptw_tw1 --n 20 --mode sampling
```

---

### Bước 4: Trực quan hóa tuyến đường

```bash
python scripts/visualize.py --model outputs/checkpoints/best.pt --problem cvrptw_tw1 --n 20
```

Kết quả sẽ được lưu dưới dạng file HTML tương tác trong thư mục `outputs/`.

---

## 📊 Cấu hình

### Điều chỉnh training cho máy của bạn

Mở `configs/training_config.yaml`:

```yaml
training:
  # CPU-light (mặc định)
  n_epochs: 10
  instances_per_epoch: 64
  batch_size_n20: 8
  batch_size_n50: 4

  # GPU đầy đủ (uncomment khi có GPU)
  # n_epochs: 50
  # instances_per_epoch: 1024000
  # batch_size_n20: 512
  # batch_size_n50: 128
```

### Hyperparameters chính

```yaml
model:
  d_node: 128           # Node embedding dimension
  d_vehicle: 128        # Vehicle embedding dimension
  n_heads: 8            # Số attention heads
  n_sa_layers: 3        # Số Self-Attention layers trong Node Encoder
  d_decoder_hidden: 256 # Decoder hidden dimension

training:
  lr_init: 0.0001       # Learning rate ban đầu
  lr_decay_gamma: 0.001 # Hệ số decay LR
  grad_clip: 1.0        # Gradient clipping norm
```

---

## 🧪 Chạy tests

```bash
pytest tests/ -v
```

---

## 📐 Công thức toán học

**Hàm chi phí CVRPTW:**

```
cost = Σ_k [ Σ_{(i,j) ∈ s_k} c_ij  +  α·λ(δ_ai^k)  +  β·λ(δ_bi^k) ]
```

Trong đó:
- `c_ij` — chi phí di chuyển (khoảng cách / thời gian) từ i đến j
- `δ_ai^k = max(a_i - T_ik, 0)` — độ lệch đến sớm (early deviation)
- `δ_bi^k = max(T_ik - b_i, 0)` — độ lệch đến muộn (late deviation)
- `λ(x) = x` — hàm phạt tuyến tính
- α, β — hệ số phạt, khác nhau theo từng biến thể TW

---

## 📚 Tài liệu tham khảo

- **JAMPR:** *"Joint Attention Model for Parallel Route-Construction"* — bài báo gốc về kiến trúc JAMPR
- **REINFORCE Baseline:** Kool, W., van Hoof, H., & Welling, M. (2019). *"Attention, Learn to Solve Routing Problems!"* ICLR 2019. [[arXiv]](https://arxiv.org/abs/1803.08475)
- **Solomon Benchmarks:** Solomon, M. M. (1987). *"Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints."* Operations Research.

---

## 📄 License

Dự án này được phân phối theo giấy phép **MIT**. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.
