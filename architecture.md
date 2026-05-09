# docs/architecture.md — Kiến trúc Model JAMPR

## Tổng quan luồng dữ liệu

```
INPUT (Problem Instance)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                    NODE ENCODER (Static)               │
│  x_i = (r_i, q_i) → Linear → [SA Block × 3] → ω^node │
│  Chạy 1 lần duy nhất cho mỗi instance                 │
└───────────────────────────────────────────────────────┘
        │ ω^node_i  (cached)
        ▼
┌───────────────────────────────────────────────────────┐
│              DECODING LOOP (T steps)                   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │         DYNAMIC ENCODERS (mỗi step)             │  │
│  │                                                  │  │
│  │  Tour Encoder: avg(g_s(ω^node_i)) for i∈s_k    │  │
│  │  Vehicle Encoder: g_v(v_k) + tour embedding     │  │
│  │  → ω^vehicle_k cho mỗi vehicle k                │  │
│  └─────────────────────────────────────────────────┘  │
│                    │                                   │
│                    ▼                                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │            CONTEXT ASSEMBLY                      │  │
│  │  C = [ω^graph; ω^fleet; ω^act; ω^node_0; ω^last]│  │
│  └─────────────────────────────────────────────────┘  │
│                    │                                   │
│                    ▼                                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │         JOINT ACTION SPACE                       │  │
│  │  M = {g_a(ω^vehicle_k, ω^node_i) | k∈K_act, i∈V}│  │
│  └─────────────────────────────────────────────────┘  │
│                    │                                   │
│                    ▼                                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │              DECODER                             │  │
│  │  h = MHA(C, M)                                  │  │
│  │  logits = clipped_attn(h, M) + mask             │  │
│  │  (k*, n*) = sample or argmax                    │  │
│  └─────────────────────────────────────────────────┘  │
│                    │                                   │
│                    ▼                                   │
│  UPDATE STATE: Thêm n* vào tour k*, update vehicle    │
│  Nếu vehicle về depot → activate vehicle mới          │
└───────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT: Solution S = {s_1, ..., s_K}, Log probabilities
```

---

## Class Diagram (Python)

```
JAMPRModel
├── NodeEncoder
│   ├── LinearProjection (Linear)
│   └── SABlock × 3
│       ├── MultiHeadAttention
│       ├── FeedForward
│       ├── ResidualConnection
│       └── BatchNorm1d
│
├── TourEncoder
│   └── FeedForward (g_s)
│
├── VehicleEncoder
│   └── FeedForward (g_v)
│
├── AffinityNetwork (g_a)
│   ├── W1 (Linear, no bias)
│   ├── W2 (Linear, no bias)
│   └── W3 (Linear, no bias)
│
└── Decoder
    ├── ContextProjection (Linear)
    ├── MultiHeadAttention (context → M)
    └── ClippedAttention (output logits)
```

---

## State Representation

```python
class VRPTWState:
    # Problem data (static)
    coords: Tensor        # (batch, N+1, 2)
    demands: Tensor       # (batch, N+1)
    time_windows: Tensor  # (batch, N+1, 2) — [a_i, b_i]
    service_times: Tensor # (batch, N+1)
    capacity: float

    # Solution state (dynamic — thay đổi sau mỗi step)
    tours: Tensor         # (batch, K, L) — K tours, max length L
    visited: Tensor       # (batch, N+1) — bool mask
    
    # Vehicle state (dynamic)
    vehicle_positions: Tensor    # (batch, K, 2) — current coords
    vehicle_times: Tensor        # (batch, K) — current time
    vehicle_loads: Tensor        # (batch, K) — remaining capacity
    vehicle_active: Tensor       # (batch, K) — bool: is active?
    active_vehicle_ids: Tensor   # (batch, m_con) — current active vehicles
    
    # Step counter
    step: int
    tour_lengths: Tensor  # (batch, K) — length của mỗi tour
```

---

## Tensor Shapes Cheat Sheet

| Tensor | Shape | Description |
|--------|-------|-------------|
| `node_features` | (B, N+1, 3) | coords + demand |
| `node_embeddings` | (B, N+1, d_node) | sau node encoder |
| `graph_embedding` | (B, d_node) | mean of node_embeddings |
| `vehicle_embeddings` | (B, K, d_vehicle) | sau vehicle encoder |
| `fleet_embedding` | (B, d_vehicle) | mean of vehicle_embeddings |
| `active_embedding` | (B, d_vehicle) | mean of active vehicles |
| `context` | (B, d_C) | concatenated context |
| `joint_space M` | (B, m_con*(N+1), d_M) | joint action embeddings |
| `mask` | (B, m_con*(N+1)) | bool: True = infeasible |
| `logits` | (B, m_con*(N+1)) | raw attention scores |
| `probs` | (B, m_con*(N+1)) | softmax probabilities |

B = batch_size, N = n_customers, K = n_vehicles, d_* = embedding dims

---

## Decoding Algorithm (Pseudocode)

```python
def decode(model, state, greedy=True):
    log_probs = []
    
    # Step 0: Encode nodes (static, done once)
    node_emb = model.node_encoder(state.node_features)
    
    while not state.is_done():
        # Step 1: Encode vehicles (dynamic)
        vehicle_emb = model.encode_vehicles(state, node_emb)
        
        # Step 2: Build context
        context = model.build_context(node_emb, vehicle_emb, state)
        
        # Step 3: Build joint action space
        M = model.build_joint_space(vehicle_emb, node_emb, state)
        
        # Step 4: Compute mask
        mask = model.compute_mask(state)
        
        # Step 5: Decode
        logits = model.decoder(context, M, mask)
        probs = F.softmax(logits, dim=-1)
        
        # Step 6: Select action
        if greedy:
            action_idx = probs.argmax(dim=-1)
        else:
            action_idx = Categorical(probs).sample()
        
        # Step 7: Map to (vehicle, node)
        vehicle_idx, node_idx = state.index_mapping(action_idx)
        
        # Step 8: Update state
        state.update(vehicle_idx, node_idx)
        
        # Log probability cho REINFORCE
        log_probs.append(probs[range(B), action_idx].log())
    
    return state.get_solution(), torch.stack(log_probs, dim=1)
```

---

## Active Vehicle Management

```python
# Luật kích hoạt/deactivate:
# - Tại t=0: m_con vehicles đầu tiên được activate
# - Vehicle k deactivate khi:
#   (a) Nó chọn depot → về depot, kết thúc tour
#   (b) Không còn feasible node nào (TW hoặc capacity)
# - Khi 1 vehicle deactivate → activate vehicle tiếp theo trong hàng đợi
# - Tối đa m_pre lần "premature return" (vehicle về depot sớm, còn capacity)

# Index mapping ϕ: index j → (k, n)
# j = k * (N+1) + n  →  k = j // (N+1), n = j % (N+1)
```
