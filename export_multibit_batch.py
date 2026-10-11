"""
export_multibit_batch.py
Export trọng số + 1000 ảnh (100/chữ số) cho 4 định dạng:
  mem/batch/          → ảnh INT8   (INT8 RTL + Pipeline RTL)
  mem/int4/           → trọng số INT4
  mem/int4/batch/     → ảnh INT4
  mem/fp16/           → trọng số "FP16" (thực chất INT16 fixed-point)
  mem/fp16/batch/     → ảnh "FP16" (INT16 fixed-point)
  mem/fp32/           → trọng số FP32 (IEEE-754 32-bit THẬT, không lượng tử hóa)
  mem/fp32/batch/     → ảnh FP32 (IEEE-754 32-bit THẬT, không lượng tử hóa)

Ghi chú về FP32: khác với "fp16" (chỉ là INT16 lượng tử hóa mô phỏng), bộ FP32
lưu ĐÚNG bit pattern IEEE-754 single-precision (không nhân scale, không làm
tròn/clip) — dùng làm "golden reference" để so sánh độ chính xác, hoặc cho mô
phỏng phần mềm/kiểm chứng thuật toán. Cyclone II trên DE2 KHÔNG có khối số học
dấu phẩy động cứng nên KHÔNG khuyến khích tổng hợp trực tiếp bộ FP32 này lên
RTL của board DE2 (sẽ cần lõi soft-float rất tốn LE).

Chạy: python export_multibit_batch.py --checkpoint mnist_cnn.pth
"""
import argparse, random, struct
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from train_mnist import SmallMNISTCNN

SAMPLES_PER_DIGIT = 100

# ── Quantization scales ────────────────────────────────────────────────
INPUT_SCALE_8  = 127
CONV_W_SCALE_8 = 64
FC_W_SCALE_8   = 32

INPUT_SCALE_4  = 7
CONV_W_SCALE_4 = 7
FC_W_SCALE_4   = 7

INPUT_SCALE_F  = 32767
CONV_W_SCALE_F = 256
FC_W_SCALE_F   = 256


def hex_val(v: int, bits: int) -> str:
    mask = (1 << bits) - 1
    return f"{(int(v) & mask):0{bits//4}X}"

def write_hex(path: Path, arr: np.ndarray, bits: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for v in arr.reshape(-1):
            f.write(hex_val(int(v), bits) + "\n")


def float32_to_hex(v) -> str:
    """Chuyen 1 gia tri sang chuoi hex 8 ky tu = bit pattern IEEE-754
    single-precision (32-bit) THAT, khong luong tu hoa/khong scale."""
    return format(struct.unpack(">I", struct.pack(">f", float(v)))[0], "08X")


def write_hex_fp32(path: Path, arr: np.ndarray):
    """Ghi mang gia tri float ra file hex, moi dong 8 ky tu = bit pattern
    IEEE-754 32-bit cua dung gia tri do (khong lam tron/clip nhu q8/q4/q16)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for v in arr.reshape(-1):
            f.write(float32_to_hex(v) + "\n")

def q8(arr, scale):
    return np.clip(np.round(arr * scale), -128, 127).astype(np.int8)

def q4(arr, scale):
    return np.clip(np.round(arr * scale), -8, 7).astype(np.int8)

def q16(arr, scale):
    return np.clip(np.round(arr * scale), -(2**15), 2**15-1).astype(np.int16)

def q32(arr, scale):
    return np.clip(np.round(arr * scale), -(2**31), 2**31-1).astype(np.int32)


def export_weights(model, base: Path):
    """Export trọng số cho INT4 và FP16."""
    conv_w = model.conv.weight.detach().numpy().squeeze(1).astype(np.float32)
    conv_b = model.conv.bias.detach().numpy().astype(np.float32)
    fc_w   = model.fc.weight.detach().numpy().astype(np.float32)
    fc_b   = model.fc.bias.detach().numpy().astype(np.float32)

    # ── INT4 weights ──────────────────────────────────────────────────
    d4 = base / "int4"
    write_hex(d4/"conv_w.hex", q4(conv_w, CONV_W_SCALE_4), 8)
    write_hex(d4/"conv_b.hex",
              q32(conv_b, INPUT_SCALE_4 * CONV_W_SCALE_4), 32)
    write_hex(d4/"fc_w.hex",   q4(fc_w, FC_W_SCALE_4), 8)
    write_hex(d4/"fc_b.hex",
              q32(fc_b, INPUT_SCALE_4 * FC_W_SCALE_4 * 2), 32)
    print(f"  INT4 weights → {d4}")

    # ── FP16 weights: dùng scale INT8 nhưng lưu 16-bit ─────────────────
    # Giữ scale giống INT8 (127/64/32) để CONV_SHIFT=7, FC_SHIFT=6 đúng
    # Lợi thế: acc 48-bit → không overflow như INT8's 32-bit
    df = base / "fp16"
    write_hex(df/"conv_w.hex", q16(conv_w, CONV_W_SCALE_8), 16)
    write_hex(df/"conv_b.hex",
              q32(conv_b, INPUT_SCALE_8 * CONV_W_SCALE_8), 32)
    write_hex(df/"fc_w.hex",   q16(fc_w, FC_W_SCALE_8), 16)
    write_hex(df/"fc_b.hex",
              q32(fc_b, INPUT_SCALE_8 * FC_W_SCALE_8), 32)
    print(f"  FP16 weights → {df}")

    # ── FP32 weights: IEEE-754 32-bit THAT, KHONG luong tu hoa ─────────
    # Khong nhan scale, khong lam tron/clip — luu dung gia tri float goc
    # cua model de lam golden reference / kiem chung do chinh xac.
    d32 = base / "fp32"
    write_hex_fp32(d32/"conv_w.hex", conv_w)
    write_hex_fp32(d32/"conv_b.hex", conv_b)
    write_hex_fp32(d32/"fc_w.hex",   fc_w)
    write_hex_fp32(d32/"fc_b.hex",   fc_b)
    print(f"  FP32 weights → {d32}  (IEEE-754 that, khong luong tu hoa)")

    # ── INT8 weights (cho tham khảo, dùng với cnn_accel_top.sv) ───────
    d8 = base
    write_hex(d8/"conv_w.hex", q8(conv_w, CONV_W_SCALE_8), 8)
    write_hex(d8/"conv_b.hex",
              q32(conv_b, INPUT_SCALE_8 * CONV_W_SCALE_8), 32)
    write_hex(d8/"fc_w.hex",   q8(fc_w, FC_W_SCALE_8), 8)
    write_hex(d8/"fc_b.hex",
              q32(fc_b, INPUT_SCALE_8 * FC_W_SCALE_8), 32)
    print(f"  INT8 weights → {d8}")


def export_images(test_ds, base: Path, samples: int):
    """Export ảnh batch cho cả 3 định dạng."""
    by_label = {d: [] for d in range(10)}
    for idx, (_, lbl) in enumerate(test_ds):
        by_label[int(lbl)].append(idx)

    labels_lines = []
    total = 0

    for digit in range(10):
        chosen = random.sample(by_label[digit], samples)
        for rank, ds_idx in enumerate(chosen):
            image_t, lbl = test_ds[ds_idx]
            assert int(lbl) == digit
            image_np = image_t.squeeze(0).numpy().astype(np.float32)
            fname = f"image_{digit}_{rank:02d}.hex"

            # INT8 ảnh
            write_hex(base / "batch" / fname,
                      q8(image_np, INPUT_SCALE_8), 8)
            # INT4 ảnh
            write_hex(base / "int4" / "batch" / fname,
                      q4(image_np, INPUT_SCALE_4), 8)
            # FP16 ảnh: scale INT8 lưu 16-bit
            write_hex(base / "fp16" / "batch" / fname,
                      q16(image_np, INPUT_SCALE_8), 16)
            # FP32 ảnh: IEEE-754 32-bit THAT, khong luong tu hoa
            write_hex_fp32(base / "fp32" / "batch" / fname, image_np)

            labels_lines.append(f"{fname} label={digit} ds_idx={ds_idx}")
            total += 1

        print(f"  Digit {digit}: {len(chosen)} ảnh ✓")

    # Lưu label file chung
    label_text = "\n".join(labels_lines) + "\n"
    for subdir in ["batch", "int4/batch", "fp16/batch", "fp32/batch"]:
        (base / subdir / "batch_labels.txt").write_text(label_text)

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="mnist_cnn.pth")
    parser.add_argument("--out-base",   default="../mem")
    parser.add_argument("--samples",    type=int, default=SAMPLES_PER_DIGIT)
    parser.add_argument("--seed",       type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Seed: {args.seed}")
    else:
        print("Seed: random (mỗi lần chạy cho bộ ảnh khác nhau)")

    base = Path(args.out_base)

    # Load model
    print(f"\nLoading model: {args.checkpoint}")
    model = SmallMNISTCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    # Export trọng số
    print("\n[1/2] Export trọng số...")
    export_weights(model, base)

    # Load MNIST test set
    print("\n[2/2] Export ảnh batch...")
    test_ds = datasets.MNIST(root="./data", train=False, download=True,
                             transform=transforms.ToTensor())
    print(f"  Tập test: {len(test_ds)} ảnh | Lấy {args.samples}/chữ số\n")

    total = export_images(test_ds, base, args.samples)

    # Tạo image_0.hex cho initial $readmemh trong RTL
    # (copy từ ảnh đầu tiên của batch)
    import shutil
    shutil.copy(base/"fp16"/"batch"/"image_0_00.hex", base/"fp16"/"image_0.hex")
    shutil.copy(base/"int4"/"batch"/"image_0_00.hex", base/"int4"/"image_0.hex")
    shutil.copy(base/"fp32"/"batch"/"image_0_00.hex", base/"fp32"/"image_0.hex")
    print("  image_0.hex created for FP16, INT4 and FP32 initial load")

    print(f"\n{'='*50}")
    print(f"Xong! Tổng {total} ảnh × 4 định dạng = {total*4} files")
    print(f"Cấu trúc thư mục mem/:")
    print(f"  mem/              → INT8 weights (conv_w, fc_w...)")
    print(f"  mem/batch/        → INT8 images  (→ INT8 RTL + Pipeline RTL)")
    print(f"  mem/int4/         → INT4 weights")
    print(f"  mem/int4/batch/   → INT4 images  (→ INT4 RTL)")
    print(f"  mem/fp16/         → \"FP16\" weights (thực chất INT16 fixed-point)")
    print(f"  mem/fp16/batch/   → \"FP16\" images  (→ FP16 RTL)")
    print(f"  mem/fp32/         → FP32 weights (IEEE-754 thật, khong luong tu hoa)")
    print(f"  mem/fp32/batch/   → FP32 images  (golden reference / mo phong phan mem)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()