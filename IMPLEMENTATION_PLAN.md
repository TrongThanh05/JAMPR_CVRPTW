# IMPLEMENTATION_PLAN.md — Kế hoạch implement từng bước

## TỔNG QUAN

Chia thành 7 giai đoạn, mỗi giai đoạn có checklist rõ ràng.
**Agent phải đánh dấu [x] sau khi hoàn thành mỗi task.**

---

## GIAI ĐOẠN 0: Setup môi trường (30 phút)

- [ ] Tạo `requirements.txt` với các dependencies
- [ ] Tạo `setup.py` hoặc `pyproject.toml`
- [ ] Tạo `scripts/setup_env.sh`
- [ ] Verify PyTorch hoạt động đúng với GPU/CPU
- [ ] Tạo `src/__init__.py` và tất cả `__init__.py` files

**Dependencies cần thiết:**
```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0          # for stats.ttest_rel (paired t-test)
pyyaml>=6.0
tensorboard>=2.13.0
matplotlib>=3.7.0      # for route visualization
tqdm>=4.65.0           # progress bars
pytest>=7.3.0          # testing
ortools>=9.6.0         # baseline comparison (optional)
```

---

## GIAI ĐOẠN 1: Utilities (1-2 giờ)

### 1.1 `src/utils/seed.py`
- [ ] `set_seed(seed: int)` — set random seeds cho Python/NumPy/PyTorch
- [ ] Test: seed cho reproducible results

### 1.2 `src/utils/logging_utils.py`
- [ ] `setup_logging(log_dir, experiment_name)` — configure logging
- [ ] `TBWriter` wrapper class cho TensorBoard
- [ ] `AverageMeter` class để track metrics

### 1.3 `src/utils/tensor_utils.py`
- [ ] `index_to_vehicle_node(idx, n_nodes)` — index mapping ϕ
- [ ] `compute_distance_matrix(coords)` — L2 distances
- [ ] `compute_travel_times(coords, service_times)` — bao gồm service time

### 1.4 `src/utils/time_utils.py`
- [ ] `Timer` context manager
- [ ] `ETACalculator` class

**Test file:** `tests/test_utils.py`

---

## GIAI ĐOẠN 2: Data Generation (2-3 giờ)

### 2.1 `src/data/generator.py` — CLASS: `VRPTWDataGenerator`
```python
class VRPTWDataGenerator:
    def generate_cvrp_instance(n: int, seed: Optional[int]) -> dict
    def generate_cvrptw_instance(n: int, seed: Optional[int]) -> dict
    def generate_batch(problem: str, n: int, batch_size: int) -> dict
```

- [ ] CVRP generation (theo Kool et al.)
  - Coords: Uniform [0,1]²
  - Demands: Uniform int [1,9], normalize bởi Q
- [ ] CVRP-TW generation (theo R201 Solomon)
  - Coords: Uniform [0,100]², normalize bởi 100
  - Demands: |N(15,10)| clipped [1,42], normalize bởi Q
  - TW generation theo algorithm trong PROJECT_SPEC §4.2
  - Service times: 10, normalize bởi 1000
- [ ] Test: kiểm tra TW feasibility (a_i ≤ b_i, depot TW consistent)

### 2.2 `src/data/dataset.py` — CLASS: `VRPTWDataset`
```python
class VRPTWDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str)
    def __len__(self) -> int
    def __getitem__(self, idx) -> dict
    
class OnlineVRPTWDataset(torch.utils.data.IterableDataset):
    """Sinh data on-the-fly cho training"""
    def __init__(self, generator, problem, n, n_instances)
    def __iter__(self)
```

### 2.3 `scripts/generate_data.py`
- [ ] Script sinh val/test sets và lưu ra disk
- [ ] Args: --problem, --n, --split, --size, --seed, --output

**Test file:** `tests/test_data.py`
- [ ] Test demand normalization
- [ ] Test TW feasibility
- [ ] Test batch generation shapes

---

## GIAI ĐOẠN 3: Model Architecture (4-6 giờ)

### 3.1 `src/models/attention.py`
- [ ] `SingleHeadAttention` — implement đúng formula trong math_formulas.md §1
- [ ] `MultiHeadAttention` — slice-based như paper (§2)
- [ ] Test: attention weights sum to 1, shape correct

### 3.2 `src/models/node_encoder.py` — CLASS: `NodeEncoder`
```python
class NodeEncoder(nn.Module):
    def __init__(self, input_dim: int, d_node: int, n_layers: int, n_heads: int)
    def forward(self, features: Tensor) -> Tensor
    # features: (B, N+1, input_dim)  →  output: (B, N+1, d_node)
```
- [ ] Linear projection layer
- [ ] 3 SA blocks (MHA + FF + Res + BN)
- [ ] Test: output shape (B, N+1, d_node)

### 3.3 `src/models/vehicle_encoder.py`
- [ ] `TourEncoder` (g_s): FF network
  - Input: `ω^node_i`, Output: `R^{d_vehicle/2}`
- [ ] `VehicleEncoder` (g_v): FF network
  - Input: vehicle features, Output: `R^{d_vehicle/2}`
- [ ] `encode_vehicles(state, node_emb)` function:
  - Combine g_v và g_s outputs
  - Handle trường hợp tour chưa có nodes (zero vector)
  - Output: `(B, K, d_vehicle)`

### 3.4 `src/models/affinity.py` — CLASS: `AffinityNetwork`
```python
class AffinityNetwork(nn.Module):
    def __init__(self, d_node: int, d_vehicle: int, d_M: int)
    def forward(self, vehicle_emb: Tensor, node_emb: Tensor) -> Tensor
    # vehicle_emb: (B, K_act, d_vehicle)
    # node_emb: (B, N+1, d_node)
    # output: (B, K_act*(N+1), d_M)
```
- [ ] Implement g_a formula từ math_formulas.md §7
- [ ] Element-wise product: `vehicle_emb ⊙ node_emb`
- [ ] Dot product: `vehicle_emb · node_emb`
- [ ] Concatenate và project qua W3

### 3.5 `src/models/decoder.py` — CLASS: `JAMPRDecoder`
```python
class JAMPRDecoder(nn.Module):
    def __init__(self, d_context: int, d_M: int, d_hidden: int, clip_value: float)
    def forward(self, context: Tensor, M: Tensor, mask: Tensor) -> Tensor
    # context: (B, d_C)
    # M: (B, seq_len, d_M)
    # mask: (B, seq_len) — True = infeasible
    # output: (B, seq_len) — logits
```
- [ ] MHA(context, M) — attend context over M
- [ ] Clipped attention: `10 * tanh(score)`
- [ ] Apply mask (set -inf)
- [ ] Return logits (không softmax ở đây)

### 3.6 `src/models/jampr.py` — CLASS: `JAMPRModel` (Main)
```python
class JAMPRModel(nn.Module):
    def __init__(self, config: ModelConfig, problem_config: dict)
    
    def forward(self, batch: dict, greedy: bool = True) -> tuple[Tensor, Tensor]
    # Returns: (solutions, log_probs)
    
    def decode_step(self, state: VRPTWState, node_emb: Tensor) -> tuple
    
    def compute_mask(self, state: VRPTWState) -> Tensor
    
    def build_context(self, node_emb, vehicle_emb, state) -> Tensor
    
    def build_joint_space(self, vehicle_emb, node_emb, state) -> Tensor
```
- [ ] Full decoding loop
- [ ] State management (VRPTWState class)
- [ ] Masking logic (đọc kỹ §9 trong math_formulas.md)
- [ ] Active vehicle management
- [ ] Premature return logic (m_pre)

**Test file:** `tests/test_models.py`
- [ ] Shape tests cho mỗi module
- [ ] Gradient flow test
- [ ] Masking correctness test
- [ ] Full forward pass test

---

## GIAI ĐOẠN 4: Training (3-4 giờ)

### 4.1 `src/training/reinforce.py`
```python
def compute_reinforce_loss(log_probs: Tensor, costs: Tensor, 
                            baseline_costs: Tensor) -> Tensor:
    """
    REINFORCE loss với baseline.
    log_probs: (B, T) — log prob của mỗi step
    costs: (B,) — total cost của solutions
    baseline_costs: (B,) — cost của baseline solutions
    """
```
- [ ] Implement REINFORCE formula từ math_formulas.md §11
- [ ] Advantage = cost - baseline (detach baseline)

### 4.2 `src/training/baseline.py` — CLASS: `RolloutBaseline`
```python
class RolloutBaseline:
    def __init__(self, model, config)
    def eval(self, batch) -> Tensor     # greedy rollout, return costs
    def update(self, model, val_data)   # update nếu model significantly better
    def wrap_dataset(self, dataset)     # đính kèm baseline costs vào dataset
```
- [ ] Greedy rollout baseline (copy weights của best model)
- [ ] Paired t-test để check significance
- [ ] Warm-up với exponential moving average

### 4.3 `src/training/scheduler.py`
```python
class InverseLRScheduler:
    """η_t = (1/(1+γ·t)) · η_{t-1}"""
    def __init__(self, optimizer, gamma: float)
    def step(self, epoch: int)
    def get_lr(self) -> float
```

### 4.4 `src/training/trainer.py` — CLASS: `Trainer`
```python
class Trainer:
    def __init__(self, model, config, train_generator, val_dataset)
    def train(self) -> None
    def train_epoch(self, epoch: int) -> dict
    def validate(self) -> dict
    def save_checkpoint(self, epoch: int, metrics: dict)
    def load_checkpoint(self, path: str)
```

**Training loop chi tiết:**
```python
for epoch in range(n_epochs):
    model.train()
    for batch in train_loader:
        # 1. Forward pass (sampling mode)
        solutions, log_probs = model(batch, greedy=False)
        costs = compute_costs(solutions, batch, problem_type)
        
        # 2. Baseline
        with torch.no_grad():
            baseline_solutions, _ = baseline.eval(batch)
            baseline_costs = compute_costs(baseline_solutions, batch, problem_type)
        
        # 3. Loss
        loss = compute_reinforce_loss(log_probs, costs, baseline_costs)
        
        # 4. Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    
    # 5. LR decay
    scheduler.step(epoch)
    
    # 6. Validate & update baseline
    val_cost = validate(model, val_dataset)
    baseline.update(model, val_dataset)
    
    # 7. Save checkpoint
    save_checkpoint(epoch, val_cost)
```

### 4.5 `src/evaluation/metrics.py`
```python
def compute_cost(tours: list, coords: Tensor, demands: Tensor,
                 time_windows: Tensor, service_times: Tensor,
                 problem_type: str, alpha: float, beta: float) -> Tensor:
    """Compute cost theo formula trong PROJECT_SPEC §1.3"""

def check_feasibility(tours, ...) -> tuple[bool, list[str]]:
    """Check tất cả constraints, return (feasible, violations)"""

def compute_arrival_times(tours, coords, service_times) -> list[list[float]]:
    """Compute arrival time tại mỗi node trong mỗi tour"""
```

**Test file:** `tests/test_training.py`
- [ ] Test REINFORCE loss computation
- [ ] Test baseline update logic
- [ ] Test LR scheduler values

---

## GIAI ĐOẠN 5: Evaluation & Visualization (2 giờ)

### 5.1 `src/evaluation/evaluator.py`
```python
class Evaluator:
    def __init__(self, model, config)
    def evaluate_greedy(self, dataset) -> dict
    def evaluate_sampling(self, dataset, n_samples: int = 1280) -> dict
    def benchmark(self, dataset) -> pd.DataFrame  # Table like paper Table 1
```

### 5.2 `src/evaluation/visualizer.py`
```python
def plot_routes(instance: dict, solution: dict, title: str, save_path: str)
"""Plot routes như Figure 2, 3 trong paper"""

def plot_learning_curves(log_dir: str, save_path: str)
"""Plot training curves như Figure 8, 9"""
```

---

## GIAI ĐOẠN 6: Scripts (1 giờ)

### 6.1 `scripts/train.py`
```python
"""
Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --debug
    python scripts/train.py --config configs/training_config.yaml --resume outputs/checkpoints/best.pt
"""
```

### 6.2 `scripts/evaluate.py`
```python
"""
Usage:
    python scripts/evaluate.py --model outputs/checkpoints/best.pt \
                                --problem cvrptw_tw1 --n 20
"""
```

### 6.3 `scripts/generate_data.py`
```python
"""
Usage:
    python scripts/generate_data.py --problem cvrptw --n 20 --split val --size 10000
"""
```

---

## GIAI ĐOẠN 7: Testing & Documentation (1-2 giờ)

- [ ] `tests/test_utils.py` — test utilities
- [ ] `tests/test_data.py` — test data generation
- [ ] `tests/test_models.py` — test model components
- [ ] `tests/test_training.py` — test training components
- [ ] `tests/test_metrics.py` — test cost computation
- [ ] `notebooks/demo.ipynb` — demo notebook
- [ ] `README.md` — installation & usage guide

---

## CHECKLIST HOÀN THÀNH

Trước khi submit, verify:
- [ ] `pytest tests/` — tất cả tests pass
- [ ] `python scripts/train.py --debug` — training loop chạy được 1 epoch
- [ ] `python scripts/evaluate.py` — evaluate script chạy được
- [ ] Model forward pass không có NaN/Inf
- [ ] Masking logic đúng (không chọn infeasible actions)
- [ ] Costs được compute đúng (verify bằng tay trên 1 ví dụ nhỏ)
- [ ] Checkpoints save/load hoạt động
- [ ] TensorBoard logs được ghi
