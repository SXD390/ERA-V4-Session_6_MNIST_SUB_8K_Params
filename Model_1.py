"""
TARGET:
- Stand up a strong baseline; validate pipeline/scheduler/BN ordering.
- Aim high accuracy fast (>=99.2%), not focused on parameter budget yet.

RESULT (from my runs):
- Params: ~13,808
- Best Test Acc (15 epochs): 99.5–99.6%
- Train Acc: ~99.4–99.6%

ANALYSIS:
- Accurate but over the 8k budget. Early padding=0 shrinks spatial maps aggressively yet generalizes well.
- Next step: switch to GAP + 1x1 transitions; tighten channels to reduce params while preserving RF and accuracy.
"""

import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self, num_classes: int = 10, p_drop: float = 0.05):
        super().__init__()
        # 28 -> 26 -> 24 (pad=0 early)
        self.c1 = nn.Conv2d(1, 16, 3, padding=0, bias=False)
        self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=0, bias=False)
        self.b2 = nn.BatchNorm2d(32)
        self.t1 = nn.Conv2d(32, 10, 1, bias=False)   # channel squeeze
        self.p1 = nn.MaxPool2d(2, 2)                 # 24->12
        self.d1 = nn.Dropout(p_drop)

        # 12 -> 10 -> 8 -> 6 (pad=0 convs grow RF fast)
        self.c3 = nn.Conv2d(10, 16, 3, padding=0, bias=False)
        self.b3 = nn.BatchNorm2d(16)
        self.c4 = nn.Conv2d(16, 16, 3, padding=0, bias=False)
        self.b4 = nn.BatchNorm2d(16)
        self.c5 = nn.Conv2d(16, 16, 3, padding=0, bias=False)
        self.b5 = nn.BatchNorm2d(16)
        self.c6 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(16)
        self.d2 = nn.Dropout(p_drop)

        # AvgPool over 6x6
        self.gap = nn.AvgPool2d(6)
        self.cls = nn.Conv2d(16, num_classes, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = self.t1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b4(self.c4(x)))
        x = F.relu(self.b5(self.c5(x)))
        x = F.relu(self.b6(self.c6(x)))
        x = self.d2(x)

        x = self.gap(x)
        x = self.cls(x).squeeze(-1).squeeze(-1)  # logits
        return x