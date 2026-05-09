# 🤖 AGENT INSTRUCTIONS — JAMPR VRPTW Project

> **Dành cho:** Claude Opus (hoặc AI agent tương đương)  
> **Mục tiêu:** Implement toàn bộ hệ thống JAMPR để giải VRPTW từ đầu đến cuối  
> **Ngôn ngữ code:** Python 3.10+, PyTorch 2.x

---

## 📋 TỔNG QUAN DỰ ÁN

Dự án này implement **JAMPR (Joint Attention Model for Parallel Route-Construction)** — một mô hình deep learning giải bài toán **VRPTW (Vehicle Routing Problem with Time Windows)**. Mô hình học cách phân công lịch trình tối ưu cho đội xe phục vụ khách hàng trong khung thời gian cho phép, sử dụng cơ chế attention để xây dựng nhiều tuyến đường song song.

**Paper gốc:** *"Learning to Solve Vehicle Routing Problems with Time Windows through Joint Attention"* — Falkner & Schmidt-Thieme (2020), arXiv:2006.09100

---

## 🗂️ CẤU TRÚC FILE DỰ ÁN

```
jampr_vrptw/
├── AGENT_INSTRUCTIONS.md       ← File này (đọc trước)
├── PROJECT_SPEC.md             ← Đặc tả kỹ thuật chi tiết
├── CODING_GUIDELINES.md        ← Quy tắc code phải tuân theo
├── IMPLEMENTATION_PLAN.md      ← Kế hoạch implement từng bước
│
├── docs/
│   ├── architecture.md         ← Kiến trúc model chi tiết
│   ├── math_formulas.md        ← Các công thức toán học
│   ├── data_format.md          ← Định dạng dữ liệu
│   └── experiment_setup.md     ← Setup thí nghiệm
│
├── configs/
│   ├── model_config.yaml       ← Hyperparameters model
│   ├── training_config.yaml    ← Config training
│   └── data_config.yaml        ← Config data generation
│
├── src/
│   ├── models/                 ← Các module model
│   ├── data/                   ← Data generation & loading
│   ├── training/               ← Training loop & baselines
│   ├── evaluation/             ← Evaluation & metrics
│   └── utils/                  ← Utilities
│
├── scripts/
│   ├── train.py                ← Script chạy training
│   ├── evaluate.py             ← Script chạy evaluation
│   └── generate_data.py        ← Script sinh dữ liệu
│
├── tests/                      ← Unit tests
└── notebooks/                  ← Jupyter notebooks
```

---

## 🚀 THỨ TỰ THỰC HIỆN (BẮT BUỘC)

Agent PHẢI thực hiện theo đúng thứ tự sau. **Không được bỏ qua bước nào.**

### PHASE 1 — Đọc hiểu (Không code)
1. Đọc `PROJECT_SPEC.md` — nắm toàn bộ yêu cầu
2. Đọc `docs/architecture.md` — hiểu kiến trúc model
3. Đọc `docs/math_formulas.md` — hiểu các công thức
4. Đọc `CODING_GUIDELINES.md` — nắm quy tắc code
5. Đọc `IMPLEMENTATION_PLAN.md` — lập kế hoạch

### PHASE 2 — Setup môi trường
```bash
# Chạy script này trước khi code bất cứ thứ gì
bash scripts/setup_env.sh
```

### PHASE 3 — Implement theo thứ tự module
```
Bước 1: src/utils/          → Utilities cơ bản
Bước 2: src/data/           → Data generation
Bước 3: src/models/         → Model architecture
Bước 4: src/training/       → Training loop
Bước 5: src/evaluation/     → Evaluation
Bước 6: scripts/            → Entry point scripts
Bước 7: tests/              → Unit tests
```

### PHASE 4 — Chạy & Kiểm tra
```bash
# Test từng module
python -m pytest tests/ -v

# Chạy quick sanity check
python scripts/train.py --config configs/training_config.yaml --debug

# Chạy full training
python scripts/train.py --config configs/training_config.yaml
```

---

## ⚠️ NGUYÊN TẮC QUAN TRỌNG

1. **Luôn đọc spec trước khi code** — Không tự suy đoán yêu cầu
2. **Implement đúng công thức trong paper** — Xem `docs/math_formulas.md`
3. **Chạy test sau mỗi module** — Không để lỗi tích lũy
4. **Comment đầy đủ bằng tiếng Anh** — Mọi function/class đều có docstring
5. **Không hardcode** — Mọi hyperparameter đều đọc từ config file
6. **Log đầy đủ** — Dùng Python `logging`, không dùng `print`
7. **Reproducibility** — Set random seed ở mọi nơi

---

## 📌 KẾT QUẢ KỲ VỌNG (Benchmark)

Sau khi implement xong, model phải đạt được kết quả tương đương paper:

| Problem | N | Model | Cost |
|---------|---|-------|------|
| TW1 | 20 | JAMPR (sampl.) | ~1716 |
| TW1 | 50 | JAMPR (sampl.) | ~2691 |
| TW2 | 20 | JAMPR (sampl.) | ~620 |
| TW2 | 50 | JAMPR (sampl.) | ~1116 |
| TW3 | 20 | JAMPR (sampl.) | ~844 |
| TW3 | 50 | JAMPR (sampl.) | ~1947 |

---

## 🆘 KHI GẶP VẤN ĐỀ

- Nếu không hiểu công thức → Xem `docs/math_formulas.md` section tương ứng
- Nếu không hiểu data format → Xem `docs/data_format.md`
- Nếu shape tensor sai → Thêm `assert` statements để debug
- Nếu loss không giảm → Kiểm tra learning rate và masking logic
