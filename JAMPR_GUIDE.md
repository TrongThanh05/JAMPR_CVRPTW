# 🗺️ Hướng Dẫn Toàn Diện: JAMPR VRPTW

## Phần 1 — Kiến Trúc Thuật Toán

### 1.1 Tổng quan bài toán

JAMPR (Joint Action Multi-vehicle Policy for Routing) giải bài toán **CVRP-TW** (Capacitated Vehicle Routing Problem with Time Windows):

| Thành phần | Mô tả |
|---|---|
| **Input** | N khách hàng + 1 depot, tọa độ, demand, time window [a_i, b_i] |
| **Output** | Tập các tour tối ưu: ít xe nhất + tổng quãng đường ngắn nhất |
| **Ràng buộc** | Mỗi xe không vượt capacity Q; mỗi khách chỉ được phục vụ 1 lần |

---

### 1.2 Kiến trúc Model — 5 Module chính

```
INPUT (batch of problem instances)
        │
        ▼
┌─────────────────────────────────────────────┐
│  1. NODE ENCODER (Static — chạy 1 lần)      │
│  x_i=(tọa độ, demand) → Linear → SA×3       │
│  Output: ω^node_i  shape=(B, N+1, 128)       │
└─────────────────────────────────────────────┘
        │ cached (không tính lại trong loop)
        ▼
┌─────────────────────────────────────────────┐
│  2. DECODING LOOP  (T bước lặp)             │
│                                              │
│  ╔═══════════════════════════════════════╗  │
│  ║  VEHICLE ENCODER (dynamic, mỗi bước) ║  │
│  ║  v_k=(id, dist_to_depot, pos, time)  ║  │
│  ║  g_v(v_k) ⊕ avg(g_s(ω^node_i∈s_k))  ║  │
│  ║  → ω^vehicle_k  shape=(B, K, 128)    ║  │
│  ╚═══════════════════════════════════════╝  │
│                   │                          │
│  ╔═══════════════════════════════════════╗  │
│  ║       CONTEXT ASSEMBLY               ║  │
│  ║  C = [ω^graph; ω^fleet; ω^active;   ║  │
│  ║        ω^depot; ω^last_node]         ║  │
│  ║  dim: 3×128 + 2×128 = 640            ║  │
│  ╚═══════════════════════════════════════╝  │
│                   │                          │
│  ╔═══════════════════════════════════════╗  │
│  ║    JOINT ACTION SPACE                ║  │
│  ║  M = {g_a(ω^vehicle_k, ω^node_i)}   ║  │
│  ║  size: m_con × (N+1) actions         ║  │
│  ╚═══════════════════════════════════════╝  │
│                   │                          │
│  ╔═══════════════════════════════════════╗  │
│  ║        DECODER                       ║  │
│  ║  h = MHA(C, M)  # attention          ║  │
│  ║  logits = clipped_attn(h, M) + mask  ║  │
│  ║  action = sample hoặc argmax         ║  │
│  ╚═══════════════════════════════════════╝  │
│                   │                          │
│  → UPDATE STATE: thêm node vào tour xe k    │
└─────────────────────────────────────────────┘
        │
        ▼
OUTPUT: Solutions S = {s_1…s_K}, Log probabilities
```

---

### 1.3 Chi tiết từng module

#### 🔷 Node Encoder (Static)
- **Input**: `x_i = (x, y, demand)` → shape `(B, N+1, 3)`
- **Xử lý**: Linear projection → 3 lớp Self-Attention Block
- **Mỗi SA Block**: MHA (8 heads) → FeedForward → Residual → BatchNorm
- **Output**: `ω^node` shape `(B, N+1, 128)`
- **Lưu ý**: Chỉ tính **1 lần** cho cả episode → tiết kiệm compute

#### 🔷 Tour Encoder (Dynamic)
- **Input**: Embeddings của các nodes đã thăm trong tour s_k
- **Mạng**: 2 hidden layers, FeedForward, output dim = 64
- **Output**: Average embedding của tour `avg(g_s(ω^node_i))`

#### 🔷 Vehicle Encoder (Dynamic)
- **Input**: `v_k = (vehicle_id, return_cost, position, current_time)` → `R^4`
- **Ghép**: `ω^vehicle_k = [g_v(v_k) ; avg_tour_embedding]` → dim = 128
- **CVRP-TW**: 3 hidden layers (phức tạp hơn CVRP: 1 layer)

#### 🔷 Affinity Network g_a (Joint Action Space)
```
g_a(ω^vehicle_k, ω^node_i) = W1·ω^node + W2·ω^vehicle 
                             + W3·[ω^vehicle ⊙ ω^node; (ω^vehicle)ᵀ·ω^node]
```
- Tạo ra **m_con × (N+1)** action candidates (xe × node)
- Đây là "joint" — mỗi bước chọn đồng thời xe NÀO và node NÀO

#### 🔷 Decoder
1. **MHA**: `h = MHA(context C, joint space M)` — query=C, key/value=M
2. **Masking**: Các action infeasible (sai TW, vượt capacity, đã thăm) → `-inf`
3. **ClippedAttention**: `logits = 10·tanh(h·M^T / √d)` — clip [-10, 10]
4. **Chọn action**: `sample()` khi train, `argmax()` khi inference

---

### 1.4 Training với REINFORCE

```
Mỗi batch:
  1. Forward pass (sampling) → solution S, log P(S)
  2. Tính cost(S) = tổng quãng đường + penalty TW
  3. Tính baseline cost(S_greedy) của model tốt nhất
  4. Loss = mean[(cost - baseline) × (-log P)]
  5. Backward + clip gradient (norm=1.0)
  6. Adam step
```

**Rollout Baseline** (Kool et al. 2019):
- Greedy rollout của **best model checkpoint**
- Paired t-test (α=0.05): nếu current model tốt hơn → cập nhật baseline
- Warm-up epoch 1: dùng exponential moving average (β=0.8)

**LR Decay** (Inverse decay): `η_t = η_0 / (1 + γ·t)`, γ=0.001

---

## Phần 2 — Quy Trình Chạy & Training

### 2.1 Cài đặt môi trường

```powershell
# Bước 1: Tạo virtual environment
cd f:\bai_tap_lap_trinh\Python\JAMPR_FOR_VRPTW
python -m venv venv
.\venv\Scripts\activate

# Bước 2: Cài dependencies
pip install -r requirements.txt

# Bước 3: Cài package ở development mode
pip install -e .
```

---

### 2.2 Bộ dữ liệu phù hợp với từng thiết bị

> Dữ liệu được **sinh ngẫu nhiên on-the-fly** — không cần tải file ngoài.
> Chỉ cần sinh validation set một lần và lưu disk.

#### 📊 Bảng cấu hình theo năng lực máy

| Thiết bị | Problem | N | instances/epoch | batch_size | Thời gian/epoch ước tính |
|---|---|---|---|---|---|
| **CPU (laptop bình thường)** | cvrptw_tw1 | 20 | 64 | 8 | ~2–5 phút |
| **CPU (máy mạnh, 8+ core)** | cvrptw_tw1 | 20 | 256 | 16 | ~5–10 phút |
| **GPU (4GB VRAM)** | cvrptw_tw1 | 20 | 4096 | 64 | ~3–5 phút |
| **GPU (8GB+ VRAM)** | cvrptw_tw1 | 20 | 16384 | 128 | ~3–5 phút |
| **GPU (12GB+ VRAM)** | cvrptw_tw1 | 50 | 4096 | 64 | ~5–8 phút |

> [!NOTE]
> Paper gốc dùng instances_per_epoch=1,024,000 với batch=512 trên GPU mạnh.
> Với CPU, configs hiện tại (64 instances, batch=8) đã được điều chỉnh phù hợp.

---

### 2.3 Step-by-step: Sinh dữ liệu validation

```powershell
# Sinh validation set nhỏ cho CPU (100 instances)
python scripts/generate_data.py `
    --problem cvrptw_tw1 `
    --n 20 `
    --split val `
    --size 100 `
    --seed 5678

# Kết quả: outputs/data/cvrptw_tw1_n20_val.pt
```

---

### 2.4 Step-by-step: Training

#### 🟢 Mode Debug (chạy vài giây, kiểm tra code không bị lỗi)
```powershell
python scripts/train.py `
    --config configs/training_config.yaml `
    --problem cvrptw_tw1 `
    --n 20 `
    --debug
```
Config debug: 2 epochs × 8 instances × batch=4 → ~10–30 giây

#### 🟡 Mode CPU-Light (training thực sự, thiết bị không có GPU)
```powershell
python scripts/train.py `
    --config configs/training_config.yaml `
    --problem cvrptw_tw1 `
    --n 20
```
Config mặc định: 10 epochs × 64 instances × batch=8

#### 🔵 Tiếp tục từ checkpoint
```powershell
python scripts/train.py `
    --config configs/training_config.yaml `
    --problem cvrptw_tw1 `
    --n 20 `
    --resume outputs/checkpoints/cvrptw_tw1_20_mcon3_epoch002.pt
```

---

### 2.5 Điều chỉnh config cho thiết bị của bạn

Mở `configs/training_config.yaml` và chỉnh các tham số:

```yaml
training:
  # ===== CPU bình thường (mặc định) =====
  n_epochs: 10
  instances_per_epoch: 64
  batch_size_n20: 8
  batch_size_n50: 4

  # ===== CPU mạnh (8+ cores, 16GB RAM) =====
  # n_epochs: 20
  # instances_per_epoch: 256
  # batch_size_n20: 16

  # ===== GPU 4–8GB =====
  # n_epochs: 50
  # instances_per_epoch: 4096
  # batch_size_n20: 64
  # batch_size_n50: 32

  # ===== GPU mạnh (12GB+) – gần paper gốc =====
  # n_epochs: 50
  # instances_per_epoch: 102400
  # batch_size_n20: 256
  # batch_size_n50: 128
```

---

### 2.6 Theo dõi kết quả training

#### Console log mỗi batch:
```
12:30:01 | INFO | Epoch 1/10 | cost=8.2341 | val=9.1052 | lr=1.000e-04
12:30:05 | INFO |   Batch 8/8 | cost=8.1234 | loss=0.0312
```

#### TensorBoard (xem đồ thị):
```powershell
tensorboard --logdir outputs/logs
# Mở browser: http://localhost:6006
```

#### Checkpoints được lưu tại:
```
outputs/checkpoints/
  cvrptw_tw1_20_mcon3_epoch002.pt
  cvrptw_tw1_20_mcon3_epoch004.pt
  ...
```

---

### 2.7 Evaluation sau khi train

```powershell
python scripts/evaluate.py `
    --checkpoint outputs/checkpoints/cvrptw_tw1_20_mcon3_epoch010.pt `
    --problem cvrptw_tw1 `
    --n 20 `
    --mode greedy
```

---

## Phần 3 — Luồng Dữ Liệu Trong 1 Training Step

```
batch = {
  coords:       (8, 21, 2)   # 8 instances, 21 nodes (1 depot + 20 customers), 2D coords
  demands:      (8, 21)
  time_windows: (8, 21, 2)   # [a_i, b_i]
  service_times:(8, 21)
  capacity:     scalar
}
        │
        ▼ model.forward(batch, greedy=False)
        │
        ├─→ node_encoder(node_features (8,21,3))
        │         └─→ node_embeddings (8, 21, 128)
        │
        ├─→ LOOP mỗi step t:
        │     ├─→ vehicle_encoder → vehicle_emb (8, K, 128)
        │     ├─→ build_context  → C (8, 640)
        │     ├─→ build_joint_space → M (8, m_con×21, 128)
        │     ├─→ compute_mask  → mask (8, m_con×21)  boolean
        │     ├─→ decoder(C, M, mask) → probs (8, m_con×21)
        │     └─→ sample action → update state
        │
        ├─→ solutions (list of tours per instance)
        └─→ log_probs (8, T)  ← dùng cho REINFORCE loss
        │
        ▼
costs = compute_cost(solutions, batch)  → (8,)
baseline_costs = baseline.eval(batch)   → (8,)
loss = mean[(costs - baseline) × -log_probs.sum(dim=1)]
        │
        ▼
loss.backward() → clip_grad_norm_ → optimizer.step()
```

---

## Phần 4 — Tóm Tắt Hyperparameters

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `d_node` | 128 | Node embedding dimension |
| `d_vehicle` | 128 | Vehicle embedding dimension |
| `n_heads` | 8 | Attention heads |
| `n_sa_layers` | 3 | Self-attention layers trong NodeEncoder |
| `m_con` | 1–4 | Số xe active đồng thời (tuned per problem) |
| `lr_init` | 1e-4 | Learning rate ban đầu |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `m_pre` | 6 | Số lần tối đa xe về depot sớm |

> [!TIP]
> Bắt đầu với `--debug` để đảm bảo code chạy không lỗi, sau đó dùng CPU-Light config để xem model có học được không (cost giảm qua các epoch). Chỉ scale lên GPU config khi đã verify stable training.

> [!WARNING]
> Nếu thấy `loss is nan, skipping update` trong log, tức là gradient bị NaN. Thử giảm `batch_size` xuống còn 4 hoặc 2 và kiểm tra lại.
