"""
export_batch.py
Export 50 ảnh ngẫu nhiên cho mỗi chữ số (0-9) từ tập MNIST test.
Tạo ra:
  - batch_images/image_DIGIT_IDX.hex   (500 file ảnh)
  - batch_labels.txt                   (nhãn thật của từng ảnh)

Chạy từ thư mục python/:
    python export_batch.py --checkpoint mnist_cnn.pth --out-dir ../mem/batch
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from train_mnist import SmallMNISTCNN
from infer_fixed_reference import (
    INPUT_SCALE,
    CONV_W_SCALE,
    CONV_OUT_SCALE,
    FC_W_SCALE,
    quantize_to_int8,
)

SAMPLES_PER_DIGIT = 100
SEED = 42


def to_hex_twos(value: int, bits: int) -> str:
    mask = (1 << bits) - 1
    return f"{(value & mask):0{bits // 4}X}"


def write_hex_file(path: Path, values: np.ndarray, bits: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in values.reshape(-1):
            f.write(to_hex_twos(int(v), bits) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="mnist_cnn.pth")
    parser.add_argument("--out-dir",    type=str, default="../mem/batch")
    parser.add_argument("--samples",    type=int, default=SAMPLES_PER_DIGIT)
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model để lấy quant params (không cần inference ở đây)
    model = SmallMNISTCNN()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Load MNIST test set
    test_ds = datasets.MNIST(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor()
    )

    # Nhóm index theo label
    label_to_indices: dict[int, list[int]] = {d: [] for d in range(10)}
    for idx, (_, label) in enumerate(test_ds):
        label_to_indices[label].append(idx)

    labels_lines = []
    total = 0

    for digit in range(10):
        # Chọn ngẫu nhiên 50 mẫu cho digit này
        chosen = random.sample(label_to_indices[digit], args.samples)
        for rank, ds_idx in enumerate(chosen):
            image_t, label = test_ds[ds_idx]
            assert label == digit

            image_np = image_t.squeeze(0).numpy().astype(np.float32)
            image_q  = quantize_to_int8(image_np, INPUT_SCALE)

            fname = f"image_{digit}_{rank:02d}.hex"
            write_hex_file(out_dir / fname, image_q, 8)
            labels_lines.append(f"{fname} label={label} ds_idx={ds_idx}")
            total += 1

    (out_dir / "batch_labels.txt").write_text(
        "\n".join(labels_lines) + "\n", encoding="utf-8"
    )

    print(f"Exported {total} images to: {out_dir}")
    print(f"  {args.samples} images × 10 digits")
    print(f"  Label file: {out_dir / 'batch_labels.txt'}")


if __name__ == "__main__":
    main()
