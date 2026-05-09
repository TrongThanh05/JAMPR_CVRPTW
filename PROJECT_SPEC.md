# PROJECT_SPEC.md — Đặc tả kỹ thuật JAMPR VRPTW

## 1. BÀI TOÁN CẦN GIẢI

### 1.1 CVRP (Capacitated VRP)
- **Input:** Đồ thị G = {V, E, q, c} với N+1 nodes (1 depot + N customers)
- **Mỗi node i có:** tọa độ `r_i ∈ R²`, demand `q_i > 0`
- **K xe đồng nhất** với capacity Q
- **Output:** Tập các tour S = {s_1, ..., s_K} sao cho:
  - (1) Tất cả customers được phục vụ
  - (2) Mỗi customer chỉ được phục vụ đúng 1 lần
  - (3) Tổng demand mỗi tour ≤ Q

### 1.2 CVRP-TW (với Time Windows)
Mở rộng CVRP với:
- **Time window** [a_i, b_i] cho mỗi customer i
- **Service duration** h_i cho mỗi customer
- **Depot time window** [a_0, b_0] = planning horizon
- **Transit cost** c_ij = thời gian di chuyển (bao gồm h_i)

### 1.3 Ba biến thể TW cần hỗ trợ

| Variant | Mô tả | α | β |
|---------|-------|---|---|
| **TW1** | Hard TW — chỉ phục vụ trong [a_i, b_i], chờ nếu sớm, skip nếu muộn | 1.0 | ∞ |
| **TW2** | Soft upper bound — phạt nếu đến muộn hơn b_i | 0.0 | 0.5 |
| **TW3** | Soft cả 2 bounds — phạt cả sớm lẫn muộn | 0.1 | 0.5 |

**Hàm cost:**
```
cost = Σ_k [ Σ_{(i,j)∈s_k∪{0}} c_ij + α·λ(δ^k_ai) + β·λ(δ^k_bi) ]
```
với:
- `δ^k_ai = max(a_i - T_ik, 0)` — early deviation
- `δ^k_bi = max(T_ik - b_i, 0)` — late deviation  
- `λ(x) = x` — linear penalty

---

## 2. KIẾN TRÚC MODEL JAMPR

### 2.1 Node Encoder (Static)
- **Input:** Node features `x_i = (r_i, q_i) ∈ R³`
- **Linear projection:** `z⁰_i = W_in·x_i + b_in` → `R^{d_emb}`
- **3 lớp Self-Attention (SA):** `ω^node_i = SA(SA(SA(z⁰_i, Z⁰)))`
- Mỗi SA block = MHA (8 heads) + FF + Residual + BatchNorm
- **d_node = 128**

### 2.2 Tour Encoder (Dynamic)
- **Input:** Embeddings của các nodes đã thăm trong tour `s_k`
- **Feed-forward NN** `g_s`: `R^{d_node} → R^{d_vehicle/2}`
- **Output:** Average của g_s outputs cho các nodes đã thăm
- **2 hidden layers, d_hidden = 64**

### 2.3 Vehicle Encoder (Dynamic)
- **Input:** Vehicle features `v_k = (k, return_cost, position, current_time)`
- **Feed-forward NN** `g_v`: `R^4 → R^{d_vehicle/2}`
- **Vehicle embedding:** `ω^vehicle_k = [g_v(v_k); avg_{i∈s_k}(g_s(ω^node_i))]`
- **CVRP:** 1 hidden layer; **CVRP-TW:** 3 hidden layers
- **d_vehicle = 128**

### 2.4 Context (Comprehensive)
```python
C = [ω^graph; ω^fleet; ω^act; ω^node_0; ω^last]
# dim: d_C = 3·d_node + 2·d_vehicle
```
Trong đó:
- `ω^graph = mean(ω^node_i)` — graph-level embedding
- `ω^fleet = mean_k(ω^vehicle_k)` — toàn bộ fleet
- `ω^act = mean_{k∈K_act}(ω^vehicle_k)` — active vehicles
- `ω^node_0` — depot embedding
- `ω^last = mean_k(ω^node_{last(s_k)})` — last visited nodes

### 2.5 Joint Action Space
```python
M^(t) = { g_a(ω^vehicle_k, ω^node_i) | k ∈ K^(t)_act, i ∈ V }
# size: |M^(t)| = m_con × N + 1
```

**Affinity function g_a:**
```python
g_a(ω^vehicle_k, ω^node_i) = W1·ω^node_i + W2·ω^vehicle_k 
                             + W3·[ω^vehicle_k ⊙ ω^node_i; (ω^vehicle_k)^T·ω^node_i]
# Output dim: d_M
```

### 2.6 Decoder
```python
# Step 1: MHA over M
h = MHA(C, M)  # d_hidden = 256

# Step 2: Attention với masking
p_i = attn(h, mask(M))  # softmax với mask = -inf cho infeasible actions

# Output: (k*, n*) = ϕ(argmax(p))  # index mapping
```

---

## 3. TRAINING

### 3.1 Algorithm
- **Policy gradient:** REINFORCE với rollout baseline (Kool et al. 2019)
- **Optimizer:** Adam, lr_init = 1e-4
- **LR decay:** `η_t = (1/(1+γ·t))·η_{t-1}`, γ = 0.001
- **Gradient clipping:** norm = 1.0
- **Epochs:** 50 (CVRP-TW), 100 (CVRP)
- **Instances/epoch:** 1,024,000
- **Batch size:** 512 (N=20), 128 (N=50)

### 3.2 Rollout Baseline
- Greedy rollout của best model checkpoint
- Evaluate trên separate validation set (re-sampled mỗi epoch)
- Paired t-test với α = 0.05 để kiểm tra significance
- Warm-up: 1 epoch với exponential average (β = 0.8)

### 3.3 Concurrency Parameter m_con
- Tuned với grid search trong {1, 2, 3, 4}
- CVRP-TW1 N=20: m_con = 3 hoặc 4
- CVRP-TW3: m_con = 1
- CVRP N=20: m_con = 1

---

## 4. DATA GENERATION

### 4.1 CVRP (theo Kool et al.)
- Locations: Uniform [0,1]²
- Demands: Uniform integer [1,9]
- Capacity: Q_20 = 30, Q_50 = 40

### 4.2 CVRP-TW (theo R201 Solomon benchmark)
- **Locations:** Uniform [0, 100]²
- **Demands:** `q = clip(|N(15, 10)|, 1, 42)` (integer)
- **Depot TW:** [0, 1000]
- **Service duration:** h_i = 10 (tất cả customers)
- **Capacity:** Q_20 = 500, Q_50 = 750
- **TW generation:**
  1. `h_hat_i = ceil(d_0i) + 1` (L2 distance from depot)
  2. `a_sample = h_hat_i`, `b_sample = 1000 - h_hat_i`
  3. `a_i ~ Uniform(h_hat_i, b_sample)`
  4. `b_i = min(floor(a_i + ε·300), b_sample)` với `ε = max(|N(0,1)|, 0.01)`

---

## 5. EVALUATION

### 5.1 Inference modes
- **Greedy:** Chọn action có probability cao nhất
- **Sampling:** Lấy 1280 samples từ stochastic policy, chọn best
- **Inference time:** Trung bình thời gian giải 1 instance (BS=1)

### 5.2 Metrics
- **Primary:** Average cost trên test set (10,000 instances)
- **Secondary:** Average số vehicles k được sử dụng
- **Tertiary:** Inference time (seconds/instance)

### 5.3 Baselines cần so sánh
- **AM+TW:** Attention Model adapted cho CVRP-TW
- **GORT-AU:** Google OR-Tools với automatic selection
- **GORT-GLS:** Google OR-Tools với guided local search

---

## 6. HYPERPARAMETERS ĐẦY ĐỦ

```yaml
model:
  d_node: 128
  d_vehicle: 128
  d_M: 128
  n_heads: 8
  n_sa_layers: 3
  d_ff_hidden: 128
  d_decoder_hidden: 256
  tour_encoder_layers: 2    # CVRP-TW
  vehicle_encoder_layers: 3  # CVRP-TW (1 cho CVRP)
  m_con: 3                  # tuned per problem

training:
  lr_init: 1e-4
  lr_decay: 0.001
  grad_clip: 1.0
  n_epochs: 50              # 100 cho CVRP
  instances_per_epoch: 1024000
  batch_size_n20: 512
  batch_size_n50: 128
  baseline_warmup_epochs: 1
  baseline_beta: 0.8
  ttest_alpha: 0.05
  m_pre: 6                  # max premature returns (3 cho CVRP)

data:
  n_customers: [20, 50]
  seed_train: 1234
  seed_val: 5678
  seed_test: 9999
  val_size: 10000
  test_size: 10000
```

---

## 7. YÊU CẦU PHI CHỨC NĂNG

- **GPU support:** CUDA, MPS (Apple Silicon), CPU fallback
- **Mixed precision:** torch.cuda.amp khi có GPU
- **Checkpoint:** Save/load model state, optimizer state, epoch
- **Logging:** TensorBoard + console logging
- **Reproducibility:** Set seed cho Python, NumPy, PyTorch
- **Memory:** Efficient tensor operations, không leak memory trong training loop
- **Testing:** Coverage > 80% cho core logic
