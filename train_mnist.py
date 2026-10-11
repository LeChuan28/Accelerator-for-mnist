import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(4 * 13 * 13, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module = None) -> tuple:
    """Đánh giá model trên loader, trả về (accuracy, avg_loss).

    Nếu không truyền criterion thì avg_loss trả về là 0.0 (giữ tương thích ngược).
    """
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            if criterion is not None:
                # reduction mặc định của CrossEntropyLoss là "mean" theo batch,
                # nên nhân lại với batch size để cộng dồn đúng rồi chia cho total ở cuối.
                loss_sum += criterion(logits, labels).item() * labels.size(0)
    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1) if criterion is not None else 0.0
    return acc, avg_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="mnist_cnn.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    model     = SmallMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── Lưu lịch sử training để vẽ đồ thị ───────────────────────────────────
    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [], "test_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss  = 0.0
        correct_train = 0
        total_train   = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            preds          = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.numel()

        avg_loss  = running_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)
        test_acc, test_loss = evaluate(model, test_loader, device, criterion)

        # Ghi vào history
        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_loss,  6))
        history["train_acc"].append(round(train_acc,  6))
        history["test_acc"].append(round(test_acc,    6))
        history["test_loss"].append(round(test_loss,  6))

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | test_loss={test_loss:.4f} | "
              f"train_acc={train_acc:.4%} | test_acc={test_acc:.4%}")

    # ── Lưu checkpoint ────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved checkpoint to: {out_path}")

    # ── Lưu history JSON (cùng tên, đổi đuôi) ────────────────────────────────
    # Ví dụ: mnist_cnn.pth  →  mnist_cnn.history.json
    history_path = out_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved history   to: {history_path}")


if __name__ == "__main__":
    main()