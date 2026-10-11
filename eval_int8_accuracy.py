"""
eval_int8_accuracy.py — Đo accuracy THỰC TẾ của mô hình int8 (đúng phép
tính fixed-point mà cnn_accel_top.sv thực hiện trên board) trên toàn bộ
tập test MNIST thật (10,000 ảnh) — không cần board, không cần vẽ tay.

Mục đích: trả lời chính xác câu hỏi "lượng tử hóa int8 để vừa RAM trên
board làm giảm accuracy bao nhiêu %?" bằng cách so sánh với baseline
float (96.7%) bạn đã đo lúc train.

Yêu cầu:
    - Đã chạy train_mnist.py trước đó (để có sẵn ./data/MNIST đã tải)
    - pip install torch torchvision numpy
    - Có sẵn 4 file trọng số: conv_w.hex, conv_b.hex, fc_w.hex, fc_b.hex
      (đặt trong thư mục --weights_dir, mặc định "mem")

Cách dùng:
    # Test nhanh trên 1000 ảnh đầu (vài chục giây):
    python eval_int8_accuracy.py --weights_dir mem --limit 1000

    # Test đầy đủ trên toàn bộ 10,000 ảnh (có thể mất vài phút vì
    # dùng vòng lặp Python thuần, không tối ưu tốc độ):
    python eval_int8_accuracy.py --weights_dir mem --limit 0
"""

import argparse
import time
import numpy as np

import torch
from torchvision import datasets, transforms


# ── Đọc file trọng số .hex (giống hệt cnn_accel_top.sv $readmemh) ──────────
def read_hex8(path, signed=False):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if signed and v >= 0x80:
                v -= 0x100
            vals.append(v)
    return vals

def read_hex32(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if v >= 0x80000000:
                v -= 0x100000000
            vals.append(v)
    return vals


# ── Tham số kiến trúc (khớp cnn_accel_top.sv) ───────────────────────────────
IMAGE_W=28; IMAGE_H=28; CONV_CH=4; K=3
CONV_OUT_W=26; CONV_OUT_H=26
POOL_OUT_W=13; POOL_OUT_H=13
FC_IN=676; FC_OUT=10
CONV_SHIFT=7; FC_SHIFT=6

def img_idx(y,x): return y*IMAGE_W+x
def conv_w_idx(oc,ky,kx): return oc*(K*K)+ky*K+kx
def conv_feat_idx(oc,y,x): return oc*(CONV_OUT_H*CONV_OUT_W)+y*CONV_OUT_W+x
def pool_feat_idx(oc,y,x): return oc*(POOL_OUT_H*POOL_OUT_W)+y*POOL_OUT_W+x
def fc_w_idx(cls,idx): return cls*FC_IN+idx

def relu_sat16(v):
    if v < 0: return 0
    elif v > 32767: return 32767
    else: return v

def infer(image, conv_w, conv_b, fc_w, fc_b):
    """Mô phỏng CHÍNH XÁC phép tính fixed-point của cnn_accel_top.sv (int8)."""
    conv_feat = [0]*(CONV_CH*CONV_OUT_H*CONV_OUT_W)
    for oc in range(CONV_CH):
        for y in range(CONV_OUT_H):
            for x in range(CONV_OUT_W):
                acc = conv_b[oc]
                for ky in range(K):
                    for kx in range(K):
                        acc += image[img_idx(y+ky,x+kx)] * conv_w[conv_w_idx(oc,ky,kx)]
                conv_feat[conv_feat_idx(oc,y,x)] = relu_sat16(acc >> CONV_SHIFT)

    pool_feat = [0]*(CONV_CH*POOL_OUT_H*POOL_OUT_W)
    for oc in range(CONV_CH):
        for py in range(POOL_OUT_H):
            for px in range(POOL_OUT_W):
                pmax = -32768
                for ky in range(2):
                    for kx in range(2):
                        v = conv_feat[conv_feat_idx(oc, py*2+ky, px*2+kx)]
                        if v > pmax: pmax = v
                pool_feat[pool_feat_idx(oc,py,px)] = pmax

    fc_score = [0]*FC_OUT
    for cls in range(FC_OUT):
        acc = fc_b[cls]
        for idx in range(FC_IN):
            acc += pool_feat[idx] * fc_w[fc_w_idx(cls,idx)]
        fc_score[cls] = acc >> FC_SHIFT

    return max(range(FC_OUT), key=lambda i: fc_score[i])


def quantize_image(tensor_1x28x28):
    """Chuyển tensor MNIST chuẩn (giá trị 0.0-1.0, do transforms.ToTensor())
    thành mảng pixel 0-127 int8 — CÙNG công thức đã dùng khi tạo
    image_0_0.hex...image_9_0.hex và trong draw_and_export.py."""
    arr = (tensor_1x28x28.squeeze(0).numpy() * 255.0).astype(np.uint16)  # 0-255
    arr7 = (arr * 127 // 255).astype(np.int16)                          # 0-127
    return arr7.flatten().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", default="mem")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--limit", type=int, default=1000,
                     help="Số ảnh test tối đa (0 = toàn bộ 10000, chạy lâu hơn)")
    args = ap.parse_args()

    print("Đang nạp trọng số...")
    conv_w = read_hex8(f"{args.weights_dir}/conv_w.hex", signed=True)
    conv_b = read_hex32(f"{args.weights_dir}/conv_b.hex")
    fc_w   = read_hex8(f"{args.weights_dir}/fc_w.hex", signed=True)
    fc_b   = read_hex32(f"{args.weights_dir}/fc_b.hex")

    print("Đang nạp tập test MNIST (dùng lại dữ liệu đã tải từ train_mnist.py)...")
    test_ds = datasets.MNIST(root=args.data_dir, train=False, download=False,
                              transform=transforms.ToTensor())

    n_total = len(test_ds) if args.limit == 0 else min(args.limit, len(test_ds))
    print(f"Sẽ đánh giá trên {n_total} ảnh test...")

    correct = 0
    confusion = [[0]*10 for _ in range(10)]
    t0 = time.time()

    for i in range(n_total):
        img_tensor, label = test_ds[i]
        pixels = quantize_image(img_tensor)
        pred = infer(pixels, conv_w, conv_b, fc_w, fc_b)
        correct += int(pred == label)
        confusion[label][pred] += 1

        if (i+1) % 200 == 0 or (i+1) == n_total:
            elapsed = time.time() - t0
            acc_so_far = 100.0 * correct / (i+1)
            print(f"  [{i+1}/{n_total}]  accuracy tạm thời: {acc_so_far:.2f}%  "
                  f"({elapsed:.1f}s, ~{elapsed/(i+1)*1000:.1f} ms/ảnh)")

    acc = 100.0 * correct / n_total
    print("\n" + "="*60)
    print(f"KẾT QUẢ: accuracy int8 fixed-point (mô phỏng đúng RTL) = {correct}/{n_total} = {acc:.2f}%")
    print("="*60)

    print("\nConfusion matrix (hàng=thực tế, cột=dự đoán):")
    header = "      " + "".join(f"{i:5d}" for i in range(10))
    print(header)
    for actual in range(10):
        row_str = f"th.{actual}: " + "".join(f"{confusion[actual][p]:5d}" for p in range(10))
        print(row_str)

    print(f"\nSo với baseline float lúc train (96.7%): "
          f"chênh lệch do lượng tử hóa int8 = {96.7 - acc:+.2f} điểm %")


if __name__ == "__main__":
    main()
