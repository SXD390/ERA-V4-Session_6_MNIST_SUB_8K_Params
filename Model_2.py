"""
TARGET:
- Reduce params toward the 8k goal while keeping >=99.3% accuracy.
- Introduce GAP head + 1x1 transitions; keep BN after each conv.

RESULT (from my runs):
- Params: ~9,382
- Best Test Acc (15 epochs): ~99.25–99.30%
- Train Acc: ~99.3–99.4%

ANALYSIS:
- Much closer to budget; GAP removes dense params and aids generalization.
- Slight underfit vs Model_1; final step is to tighten channels further (<8k)
  while preserving two MaxPools, BN, and OneCycleLR schedule.
"""

import torch.nn as nn
import torch.nn.functional as F

class Model_2(nn.Module):
    def __init__(self, num_classes: int = 10, p_drop: float = 0.05):
        super().__init__()
        # Block A (pad=1 keeps spatial size); 28x28
        self.c1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(16)
        self.p1 = nn.MaxPool2d(2)        # 28->14
        self.t1 = nn.Conv2d(16, 12, 1, bias=False)
        self.d1 = nn.Dropout(p_drop)

        # Block B
        self.c3 = nn.Conv2d(12, 20, 3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(20)
        self.c4 = nn.Conv2d(20, 20, 3, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(20)
        self.p2 = nn.MaxPool2d(2)        # 14->7
        self.t2 = nn.Conv2d(20, 24, 1, bias=False)
        self.d2 = nn.Dropout(p_drop)

        # Head: GAP + 1x1 logits
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Conv2d(24, num_classes, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = self.p1(x); x = self.t1(x); x = self.d1(x)

        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b4(self.c4(x)))
        x = self.p2(x); x = self.t2(x); x = self.d2(x)

        x = self.gap(x)
        x = self.cls(x).squeeze(-1).squeeze(-1)  # logits
        return x