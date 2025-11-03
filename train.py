# one trainer for all models; OneCycleLR per-batch; MPS/CUDA
import argparse, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# import models
from model1 import Model_1
from model2 import Model_2
from model3 import Model_3

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def get_loaders(batch_train=128, batch_test=1024, device=torch.device("cpu"), seed=42):
    MEAN, STD = (0.1307,), (0.3081,)
    train_tfms = transforms.Compose([
        transforms.RandomRotation(7, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=train_tfms)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=test_tfms)

    # Generator fix for MPS
    if device.type == "mps":
        g = torch.Generator(device="mps").manual_seed(seed)
    else:
        g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True, num_workers=2,
                              pin_memory=(device.type=="cuda"), generator=g)
    test_loader  = DataLoader(test_ds, batch_size=batch_test, shuffle=False, num_workers=2,
                              pin_memory=(device.type=="cuda"))
    return train_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return (loss_sum/total), (100.0*correct/total)

def train_one(model, device, epochs=15, max_lr=0.30, label_smoothing=0.02):
    train_loader, test_loader = get_loaders(device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = optim.SGD(model.parameters(), lr=max_lr/10.0, momentum=0.9, nesterov=True, weight_decay=0.0)
    steps_per_epoch = len(train_loader)
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.2, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0
    )

    best = 0.0
    last3 = []
    for epoch in range(1, epochs+1):
        model.train()
        correct, processed, runloss = 0, 0, 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for bi, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward(); opt.step(); sched.step()

            runloss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            processed += y.size(0)
            pbar.set_description(
                f"Epoch {epoch}/{epochs} | Loss={loss.item():.4f} "
                f"Batch={bi} Acc={100*correct/processed:.2f}% LR={sched.get_last_lr()[0]:.4f}"
            )
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        best = max(best, te_acc)
        last3 = (last3 + [te_acc])[-3:]
        print(f"[{epoch:02d}] train_acc={100*correct/processed:.2f}% "
              f"test_acc={te_acc:.2f}% (best={best:.2f}%) | last3_avg={sum(last3)/len(last3):.2f}%")
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["model1", "model2", "model3", "all"], default="model3")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    device = pick_device()
    print("Device:", device)
    seed_all(42)

    def run_m(name, mcls):
        model = mcls().to(device)
        print(f"\n=== {name} ===")
        print("Params:", count_params(model))
        best = train_one(model, device=device, epochs=args.epochs)
        print(f"Best test acc: {best:.2f}%")

    if args.model in ("model1", "all"): run_m("Model_1", Model_1)
    if args.model in ("model2", "all"): run_m("Model_2", Model_2)
    if args.model in ("model3", "all"): run_m("Model_3", Model_3)

if __name__ == "__main__":
    main()