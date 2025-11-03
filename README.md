# MNIST — Session 6 (≤8K params, ≤15 epochs, ≥99.4% target)

This repo contains **three modular CNN experiments** on MNIST, implemented as `model1.py`, `model2.py`, `model3.py`, trained via `train.py`.

## Assignment Requirements 

* **Accuracy:** ≥ **99.4%** and **consistent** in the last few epochs (Model 3 is consistently ≥99.4% by the end)
* **Epochs:** ≤ **15**
* **Parameters:** ≤ **8,000** (Model 3: **7,784**)
* **Modular code:** separate model files + one train script(mainly dervied from the master .ipynb file I've been playing with)
* **Target / Result / Analysis:** present inside each model file (top docstring)
* **Receptive Field:** documented in `RF.md`
* **Clean structure:** this README + `CHANGELOG.md` + file links

---

## 📁 Structure

```
.
├── model1.py              # Model_1 (baseline, over budget)
├── model2.py              # Model_2 (GAP + transitions, tighter)
├── model3.py              # Model_3 (final sub-8k)
├── train.py               # Common trainer (CUDA / MPS / CPU)
├── README.md
├── CHANGELOG.md
└── RF.md
```

---

## 🧪 Dataset & Transforms

* **Train:** `RandomRotation(7°, fill=0)` → `ToTensor()` → `Normalize((0.1307,), (0.3081,))`
* **Test:**  `ToTensor()` → `Normalize((0.1307,), (0.3081,))`

---

## 🎯 Targets • Results • Analysis (summary)

### First Attempt — **Model 1** (`model1.py`)

* **Target:** Validate pipeline & scheduling; get strong baseline accuracy quickly.
* **Results:** ~**13,808** params; best test **99.58%** @ 15 epochs.
* **Analysis:** Accurate but exceeds 8k budget. Next: use GAP + 1×1 transitions; tighten channels.

### Second Attempt — **Model 2** (`model2.py`)

* **Target:** Reduce params toward 8k while keeping ≥99.3%.
* **Results:** ~**9,382** params; best test **99.27%**.
* **Analysis:** Close to target; slight underfit vs Model 1. Final step: sub-8k with BN + OneCycleLR. I noticed that the tree structure(Squeeze - pull) of CNN didn't work here. The slow upscaling of channels(in final model) helped.

### Third Attempt — **Model 3 (Final)** (`model3.py`)

* **Target:** ≤**8,000** params; ≤**15** epochs; **≥99.4%** consistent.
* **Results:** **7,784** params; best test **99.47%**; last epochs consistently ≥99.4%.
* **Analysis:** Compact channels, two MaxPools, BN everywhere, GAP head + Linear(18→10), OneCycleLR (cosine) + CE (label smoothing). Train/test curves tight → good generalization. **Meets the brief**.

---

## Per-Epoch Training Results

> Columns: **Epoch**, **Train Acc (%)**, **LR**, **Test Loss**, **Test Acc (%)**
> (Taken from your logs; LR is the last batch LR for that epoch.)

### Model 1 — `model1.py`

| Epoch | Train Acc (%) |     LR | Test Loss | Test Acc (%) |
| ----: | ------------: | -----: | --------: | -----------: |
|     1 |         91.82 | 0.0976 |    0.2189 |        97.82 |
|     2 |         98.20 | 0.2327 |    0.1825 |        98.92 |
|     3 |         98.47 | 0.3000 |    0.1713 |        99.29 |
|     4 |         98.71 | 0.2949 |    0.1671 |        99.37 |
|     5 |         98.85 | 0.2799 |    0.1665 |        99.24 |
|     6 |         98.89 | 0.2561 |    0.1651 |        99.32 |
|     7 |         98.94 | 0.2250 |    0.1637 |        99.31 |
|     8 |         99.04 | 0.1889 |    0.1620 |        99.31 |
|     9 |         99.09 | 0.1501 |    0.1621 |        99.27 |
|    10 |         99.12 | 0.1113 |    0.1598 |        99.45 |
|    11 |         99.17 | 0.0752 |    0.1590 |        99.37 |
|    12 |         99.23 | 0.0441 |    0.1573 |        99.46 |
|    13 |         99.34 | 0.0203 |    0.1546 |    **99.58** |
|    14 |         99.32 | 0.0054 |    0.1542 |        99.55 |
|    15 |         99.40 | 0.0003 |    0.1542 |        99.50 |

**Best test accuracy:** **99.58%**

---

### Model 2 — `model2.py`

| Epoch | Train Acc (%) |     LR | Test Loss | Test Acc (%) |
| ----: | ------------: | -----: | --------: | -----------: |
|     1 |         83.53 | 0.0976 |    0.2627 |        97.42 |
|     2 |         97.02 | 0.2327 |    0.2271 |        98.28 |
|     3 |         97.71 | 0.3000 |    0.2342 |        97.98 |
|     4 |         98.04 | 0.2949 |    0.2125 |        98.28 |
|     5 |         98.26 | 0.2799 |    0.1966 |        98.71 |
|     6 |         98.49 | 0.2561 |    0.2156 |        98.49 |
|     7 |         98.60 | 0.2250 |    0.1937 |        98.91 |
|     8 |         98.75 | 0.1889 |    0.1871 |        99.10 |
|     9 |         98.87 | 0.1501 |    0.1908 |        98.88 |
|    10 |         98.91 | 0.1113 |    0.1837 |        99.16 |
|    11 |         98.96 | 0.0752 |    0.1815 |        99.08 |
|    12 |         99.06 | 0.0441 |    0.1792 |        99.18 |
|    13 |         99.08 | 0.0203 |    0.1767 |        99.25 |
|    14 |         99.22 | 0.0054 |    0.1747 |        99.27 |
|    15 |         99.22 | 0.0003 |    0.1744 |    **99.27** |

**Best test accuracy:** **99.27%**

---

### Model 3 — `model3.py` (Final)

| Epoch | Train Acc (%) |     LR | Test Loss | Test Acc (%) |
| ----: | ------------: | -----: | --------: | -----------: |
|     1 |         92.06 | 0.0976 |    0.2190 |        97.97 |
|     2 |         98.12 | 0.2327 |    0.1878 |        98.56 |
|     3 |         98.46 | 0.3000 |    0.1794 |        98.65 |
|     4 |         98.66 | 0.2949 |    0.1736 |        98.88 |
|     5 |         98.83 | 0.2799 |    0.1628 |        99.26 |
|     6 |         98.96 | 0.2561 |    0.1632 |        99.17 |
|     7 |         99.06 | 0.2250 |    0.1643 |        99.13 |
|     8 |         99.14 | 0.1889 |    0.1584 |        99.36 |
|     9 |         99.14 | 0.1501 |    0.1631 |        99.19 |
|    10 |         99.23 | 0.1113 |    0.1579 |        99.26 |
|    11 |         99.32 | 0.0752 |    0.1556 |        99.36 |
|    12 |         99.40 | 0.0441 |    0.1533 |        99.43 |
|    13 |         99.44 | 0.0203 |    0.1518 |    **99.47** |
|    14 |         99.51 | 0.0054 |    0.1510 |    **99.47** |
|    15 |         99.58 | 0.0003 |    0.1509 |        99.44 |

**Best test accuracy:** **99.47%**
**Last-3-epoch average:** (99.47 + 99.47 + 99.44) / 3 = **99.46%** 

---

## 📊 Receptive Field

See `RF.md` for per-layer RF tables and effective RF at the logits for each model:

* Model 1: **32 px**
* Model 2: **40 px**
* Model 3: **42 px**

---

## 🔗 Files

* **Model 1:** `model1.py` (baseline, over budget)
* **Model 2:** `model2.py` (tighter, GAP + transitions)
* **Model 3:** `model3.py` (**final**, ≤8k, ≥99.4% consistent)
* **Trainer:** `train.py`
* **RF details:** `RF.md`
* **Changelog:** `CHANGELOG.md`
* **Master .ipynb file**

---

### Notes

* Device is auto-selected (CUDA → MPS → CPU).
* OneCycleLR is stepped **per batch**.
* Loss: `CrossEntropyLoss(label_smoothing=0.02)`; Optimizer: `SGD(momentum=0.9,nesterov=True)`.
