"""
TARGET:
- ≤ 8,000 parameters, ≤ 15 epochs, ≥ 99.4% test accuracy (consistent in last few epochs).

RESULT (from my runs):
- Params: 7,784
- Best Test Acc (15 epochs): ~99.47–99.50% (≥99.4% stable in final epochs)
- Train Acc: ~99.4–99.6%

ANALYSIS:
- Compact channels (1→7→10→…→18), two MaxPools to grow RF efficiently,
  BN after every conv, GAP head + Linear(18→10), OneCycleLR (cosine) + CE (label smoothing).
- Train/test curves stay tight; generalization is solid at sub-8k.
"""

import torch.nn as nn
import torch.nn.functional as F

class Model_3(nn.Module):
    def __init__(self, num_classes: int = 10, p_drop: float = 0.05):
        super().__init__()
        # Early pad=0 convs to expand RF while shrinking spatial dims (22x22 after 3 convs)
        self.c1 = nn.Conv2d(1, 7, 3, padding=0, bias=False)   # 28->26
        self.b1 = nn.BatchNorm2d(7)
        self.c2 = nn.Conv2d(7, 7, 3, padding=0, bias=False)    # 26->24
        self.b2 = nn.BatchNorm2d(7)
        self.c3 = nn.Conv2d(7, 10, 3, padding=0, bias=False)   # 24->22
        self.b3 = nn.BatchNorm2d(10)
        self.p1 = nn.MaxPool2d(2)                               # 22->11

        self.c4 = nn.Conv2d(10, 10, 3, padding=0, bias=False)   # 11->9
        self.b4 = nn.BatchNorm2d(10)
        self.c5 = nn.Conv2d(10, 12, 3, padding=0, bias=False)   # 9->7
        self.b5 = nn.BatchNorm2d(12)
        self.p2 = nn.MaxPool2d(2)                               # 7->3

        # keep spatial (3x3) with pad=1
        self.c6 = nn.Conv2d(12, 16, 3, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(16)
        self.c7 = nn.Conv2d(16, 18, 3, padding=1, bias=False)
        self.b7 = nn.BatchNorm2d(18)

        # Head: GAP + Linear
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(18, num_classes)

        self.d = nn.Dropout(p_drop)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = self.p1(x)

        x = F.relu(self.b4(self.c4(x)))
        x = F.relu(self.b5(self.c5(x)))
        x = self.p2(x)

        x = F.relu(self.b6(self.c6(x)))
        x = self.d(F.relu(self.b7(self.c7(x))))

        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)  # logits
        return x