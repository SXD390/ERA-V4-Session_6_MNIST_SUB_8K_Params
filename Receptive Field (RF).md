# Receptive Field (RF) — MNIST 28×28

Update rules:
- `rf_out = rf_in + (k - 1) * jump_in`
- `jump_out = jump_in * stride`
- 1×1 convs do not change RF; padding affects spatial size, not RF/jump.

---

## Model 1 (model1.py) — ~13,808 params

| Layer | k/s/p | H×W (comment) | RF | jump |
|---|---|---|---:|---:|
| Input | – | 28×28 | 1 | 1 |
| Conv3→16 | 3/1/0 | 26×26 | 3 | 1 |
| Conv3→32 | 3/1/0 | 24×24 | 5 | 1 |
| 1×1→10 | 1/1/0 | 24×24 | 5 | 1 |
| MaxPool2 | 2/2/0 | 12×12 | 6 | 2 |
| Conv3→16 | 3/1/0 | 10×10 | 10 | 2 |
| Conv3→16 | 3/1/0 | 8×8 | 14 | 2 |
| Conv3→16 | 3/1/0 | 6×6 | 18 | 2 |
| Conv3(pad1)→16 | 3/1/1 | 6×6 | 22 | 2 |
| AvgPool6 | 6/6/0 | 1×1 | **32** | 12 |
| 1×1→10 | 1/1/0 | 1×1 | **32** | 12 |

**Effective RF:** **32 px**

---

## Model 2 (model2.py) — ~9,382 params

| Layer | k/s/p | H×W | RF | jump |
|---|---|---|---:|---:|
| Input | – | 28×28 | 1 | 1 |
| Conv3→16 | 3/1/1 | 28×28 | 3 | 1 |
| Conv3→16 | 3/1/1 | 28×28 | 5 | 1 |
| MaxPool2 | 2/2/0 | 14×14 | 6 | 2 |
| 1×1→12 | 1/1/0 | 14×14 | 6 | 2 |
| Conv3→20 | 3/1/1 | 14×14 | 10 | 2 |
| Conv3→20 | 3/1/1 | 14×14 | 14 | 2 |
| MaxPool2 | 2/2/0 | 7×7 | 16 | 4 |
| 1×1→24 | 1/1/0 | 7×7 | 16 | 4 |
| GAP7 | 7/7/0 | 1×1 | **40** | 28 |
| 1×1→10 | 1/1/0 | 1×1 | **40** | 28 |

**Effective RF:** **40 px**

---

## Model 3 (model3.py) — 7,784 params (FINAL)

| Layer | k/s/p | H×W | RF | jump |
|---|---|---|---:|---:|
| Input | – | 28×28 | 1 | 1 |
| Conv3→7 | 3/1/0 | 26×26 | 3 | 1 |
| Conv3→7 | 3/1/0 | 24×24 | 5 | 1 |
| Conv3→10 | 3/1/0 | 22×22 | 7 | 1 |
| MaxPool2 | 2/2/0 | 11×11 | 8 | 2 |
| Conv3→10 | 3/1/0 | 9×9 | 12 | 2 |
| Conv3→12 | 3/1/0 | 7×7 | 16 | 2 |
| MaxPool2 | 2/2/0 | 3×3 | 18 | 4 |
| Conv3(p1)→16 | 3/1/1 | 3×3 | 26 | 4 |
| Conv3(p1)→18 | 3/1/1 | 3×3 | **34** | 4 |
| GAP3 | 3/3/0 | 1×1 | **42** | 12 |
| Linear18→10 | – | 1×1 | **42** | 12 |

**Effective RF:** **42 px**