# docs/math_formulas.md — Công thức toán học JAMPR

## 1. SINGLE-HEAD ATTENTION (SHA)

```
SHA(z, Z; W_q, W_k, W_v) = Σ_j attn(z, Z)_j · W_v · Z_j

attn(z, Z; W_q, W_k) = softmax( (1/√d_key) · z^T · W_q^T · W_k · Z_j  ∀j )

Kích thước:
  W_v ∈ R^{d_out × d_in}
  W_q, W_k ∈ R^{d_key × d_in}
```

**PyTorch implementation hint:**
```python
# query shape: (batch, d_in)
# keys/values shape: (batch, seq_len, d_in)
scores = torch.matmul(query @ W_q.T, (keys @ W_k.T).transpose(-2, -1)) / sqrt(d_key)
attn_weights = F.softmax(scores + mask, dim=-1)
output = torch.matmul(attn_weights, values @ W_v.T)
```

---

## 2. MULTI-HEAD ATTENTION (MHA)

```
MHA(z, Z; W) = Σ_{h=1}^{H} W^head_h · SHA(z_slice(h), Z_{., slice(h)}; W^q_h, W^k_h, W^v_h)

slice(h) = indices cho head h = [1+h·Δh, ..., Δh+h·Δh] với Δh = d_in/H
```

**Lưu ý:** Mỗi head chỉ nhìn vào 1 slice của input (không phải toàn bộ như standard MHA).

---

## 3. SELF-ATTENTION BLOCK (SA Block)

```
z^l_i = SA(z^{l-1}_i, Z^{l-1})
       = BN( FFres( BN( MHAres(z^{l-1}_i, Z^{l-1}) ) ) )

MHAres(z_i, Z) = z_i + MHA(z_i, Z)
FFres(z_i)     = z_i + FF(z_i)
FF(z_i)        = max(0, W·z_i + b)   ← ReLU activation
BN(z_i)        = W ⊙ (z_i - μ_B)/√(σ²_B + ε) + b
```

---

## 4. NODE ENCODER

```
Input: x_i = (r_i, q_i) ∈ R³   (r_i là tọa độ 2D, q_i là demand đã normalize)

z⁰_i = W_in · x_i + b_in        ← Linear projection → R^{d_node}

ω^node_i = SA₃(SA₂(SA₁(z⁰_i, Z⁰), Z¹), Z²)   ← 3 SA blocks

ω^graph = (1/(N+1)) · Σ_{i=0}^{N} ω^node_i     ← Mean pooling (kể cả depot)
```

---

## 5. VEHICLE & TOUR ENCODER

```
Vehicle features v_k = (k_normalized, return_cost_k, pos_x_k, pos_y_k, time_k)
                     ∈ R⁵  (normalize về [0,1])

Tour embedding của tour k:
  ω^vehicle_k = [ g_v(v_k) ; (1/|s_k|) · Σ_{i∈s_k} g_s(ω^node_i) ]
              ∈ R^{d_vehicle}   (concatenation)

Với:
  g_v: FF network (d_vehicle/2 output)
  g_s: FF network (d_vehicle/2 output)
  |s_k|: số nodes đã thăm trong tour k (nếu = 0, dùng zero vector)
```

---

## 6. CONTEXT EMBEDDING

```
ω^fleet = (1/K) · Σ_{k=1}^{K} ω^vehicle_k

ω^act = (1/K_act) · Σ_{k∈K_act} ω^vehicle_k

ω^last = (1/K) · Σ_{k=1}^{K} ω^node_{last(s_k)}
         (last(s_k) = depot node 0 nếu tour chưa bắt đầu)

C = [ ω^graph ; ω^fleet ; ω^act ; ω^node_0 ; ω^last ]
  ∈ R^{d_C}   với d_C = 3·d_node + 2·d_vehicle
```

---

## 7. JOINT ACTION SPACE

```
g_a(ω^vehicle_k, ω^node_i) = W₁·ω^node_i 
                             + W₂·ω^vehicle_k 
                             + W₃·[ ω^vehicle_k ⊙ ω^node_i ; 
                                    (ω^vehicle_k)^T · ω^node_i ]

Trong đó:
  ⊙ = element-wise product
  (ω^vehicle_k)^T · ω^node_i = dot product (scalar → broadcast hoặc thêm chiều)
  
  W₁ ∈ R^{d_M × d_node}
  W₂ ∈ R^{d_M × d_vehicle}
  W₃ ∈ R^{d_M × (d_node + 1)}   ← concat [elementwise; dot] → d_node + 1 dim

M^(t) = { g_a(ω^vehicle_k, ω^node_i) | k ∈ K^(t)_act, i ∈ V }
|M^(t)| = m_con × (N+1)    ← N customers + 1 depot, cho m_con active vehicles
```

---

## 8. DECODER

```
# Bước 1: MHA context → M
h = MHA(C, M^(t))    ∈ R^{d_M}

# Bước 2: Clipped attention với mask
u_j = (10) · tanh( q^T · k_j / √d_M )   ← clipping như paper [17]
  với q = W_Q · h, k_j = W_K · m_j

u_j = -∞  nếu action (k, i) không hợp lệ (infeasible)

p_j = softmax(u_j)   → probability distribution

# Chọn action
j* = argmax(p_j)  (greedy)  hoặc  j* ~ p  (sampling)
(k*, n*) = ϕ(j*)   ← index mapping từ joint space về (vehicle, node)
```

---

## 9. MASKING RULES

Action (k, n) bị mask (= -∞) khi:
1. **Node n đã được phục vụ** (bởi bất kỳ tour nào)
2. **Vehicle k không active** (không trong K_act)
3. **[TW1] Arrival time > b_n** (vi phạm hard deadline)
4. **Capacity constraint:** remaining_cap_k < demand_n
5. **[TW1] Không thể về depot đúng hạn** sau khi phục vụ n

**Depot (node 0) luôn hợp lệ** cho active vehicle (để kết thúc tour sớm).

---

## 10. TIME COMPUTATION

```python
# Arrival time của vehicle k tại node n:
arrival_k_n = max(current_time_k + travel_time(pos_k, n), a_n)
              # chờ nếu đến sớm hơn time window

# Update time sau khi phục vụ:
new_time_k = arrival_k_n + h_n    # h_n = service duration

# Travel time (bao gồm service duration của node xuất phát):
travel_time(i, j) = ||r_i - r_j||₂ + h_i
# Lưu ý: h_i đã ĐƯỢC INCLUDE trong travel time theo paper
```

---

## 11. REINFORCE GRADIENT

```
∇_θ J(θ) ≈ (1/B) · Σ_b [ (cost(s_b) - b(x_b)) · Σ_t ∇_θ log π(a^(t)_b | ·) ]

Với:
  b(x_b) = baseline cost (greedy rollout của best model)
  cost(s_b) = total cost của solution s_b
  Ký hiệu: cost thấp hơn = tốt hơn (minimize)
  Loss = mean over batch của (cost - baseline) * log_prob (với sign âm vì maximize reward)
```

---

## 12. NORMALIZE FEATURES

```python
# Node coordinates: normalize về [0, 1]
r_i_norm = r_i / 100.0   (nếu dữ liệu trong [0, 100])

# Demands: normalize bởi capacity Q
q_i_norm = q_i / Q

# Time features: normalize bởi max time horizon
t_norm = t / b_0   (b_0 = 1000 là tổng time horizon)

# Vehicle index: normalize bởi K
k_norm = k / K
```
