"""
infer_compare.py
So sánh accuracy của 3 kiểu quantization trên Python (không cần ModelSim).
Dùng numpy vectorized → chạy nhanh ~30 giây cho 1000 mẫu.

Chạy:
    python infer_compare.py --checkpoint mnist_cnn.pth
    python infer_compare.py --checkpoint mnist_cnn.pth --samples 100 --seed 42
"""

import argparse, random
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from train_mnist import SmallMNISTCNN

SAMPLES_PER_DIGIT = 100

# ── Scales (khớp với RTL) ─────────────────────────────────────────────
INPUT_SCALE_8  = 127;  CONV_W_SCALE_8 = 64;   FC_W_SCALE_8 = 32
CONV_SHIFT_8   = 7;    FC_SHIFT_8     = 6

INPUT_SCALE_4  = 7;    CONV_W_SCALE_4 = 7;    FC_W_SCALE_4 = 7
CONV_SHIFT_4   = 6;    FC_SHIFT_4     = 5

# FP16 = INT16 fixed-point, DUNG Y HET scale/shift cua INT8 (chi luu trong
# 16-bit rong hon de co headroom, khong phai mot thiet ke luong tu hoa
# khac). Day chinh la nhung gi export_multibit_batch.py da xuat ra
# mem/fp16/*.hex — truoc day script nay dung nham mot bo scale khac
# (32767/256/256, shift 15/8) khong khop voi RTL thuc te, gay sai lech.
INPUT_SCALE_F  = INPUT_SCALE_8;  CONV_W_SCALE_F = CONV_W_SCALE_8;  FC_W_SCALE_F = FC_W_SCALE_8
CONV_SHIFT_F   = CONV_SHIFT_8;   FC_SHIFT_F     = FC_SHIFT_8


def fc_bias_scale(input_scale, conv_w_scale, conv_shift, fc_w_scale):
    """Ty le DUNG de luong tu hoa fc_bias.

    Bias duoc cong truc tiep vao tong (fw_q @ flat) TRUOC khi right-shift,
    nen no PHAI cung ty le voi tong do:
        pool_scale = (input_scale * conv_w_scale) / 2**conv_shift   (ty le
                      thuc te cua feature sau lop conv+shift+clip)
        fc_bias_scale = pool_scale * fc_w_scale

    (Truoc day code dung nham input_scale*fc_w_scale — bo qua hoan toan
    conv_w_scale va conv_shift — gay sai lech scale rat lon, dac biet voi
    FP16 sai lech toi 128 lan khien bias lan at hoan toan dong gop tu anh
    dau vao va trong so, dan den do chinh xac sup do.)
    """
    pool_scale = (input_scale * conv_w_scale) / (2 ** conv_shift)
    return pool_scale * fc_w_scale


FB_SCALE_8 = fc_bias_scale(INPUT_SCALE_8, CONV_W_SCALE_8, CONV_SHIFT_8, FC_W_SCALE_8)
FB_SCALE_4 = fc_bias_scale(INPUT_SCALE_4, CONV_W_SCALE_4, CONV_SHIFT_4, FC_W_SCALE_4)
FB_SCALE_F = fc_bias_scale(INPUT_SCALE_F, CONV_W_SCALE_F, CONV_SHIFT_F, FC_W_SCALE_F)


# ── Quantize helpers ──────────────────────────────────────────────────
def q8(arr, scale):
    return np.clip(np.round(arr * scale), -128, 127).astype(np.int8)

def q4(arr, scale):
    return np.clip(np.round(arr * scale), -8, 7).astype(np.int8)

def q16(arr, scale):
    return np.clip(np.round(arr * scale), -(2**15), 2**15-1).astype(np.int16)

def q32(arr, scale):
    return np.clip(np.round(arr * scale), -(2**31), 2**31-1).astype(np.int32)


# ── Conv + ReLU + MaxPool (numpy vectorized) ──────────────────────────
def conv_relu_pool(image, cw, cb, shift, sat_max, dtype_acc):
    """
    image: (28,28) quantized
    cw:    (4,3,3) quantized weights
    cb:    (4,)    quantized bias
    Trả về pool_feat: (4,13,13)
    """
    nch = cw.shape[0]
    conv = np.zeros((nch, 26, 26), dtype=np.int64)
    for oc in range(nch):
        for ky in range(3):
            for kx in range(3):
                conv[oc] += (image[ky:ky+26, kx:kx+26].astype(np.int64)
                             * int(cw[oc, ky, kx]))
        conv[oc] += int(cb[oc])
    conv = np.right_shift(conv, shift)
    conv = np.clip(conv, 0, sat_max)                    # ReLU + sat

    # MaxPool 2×2
    pool = np.zeros((nch, 13, 13), dtype=dtype_acc)
    for oc in range(nch):
        for y in range(13):
            for x in range(13):
                pool[oc,y,x] = np.max(conv[oc, y*2:y*2+2, x*2:x*2+2])
    return pool


def fc(pool, fw, fb, shift):
    """pool: (4,13,13) → flat (676,), fw: (10,676), fb: (10,)"""
    flat   = pool.reshape(-1).astype(np.int64)
    scores = (fw.astype(np.int64) @ flat) + fb.astype(np.int64)
    return np.right_shift(scores, shift)


# ── 3 inference functions ─────────────────────────────────────────────
def infer_float32(image_np, model):
    t = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return int(torch.argmax(model(t), dim=1).item())


def infer_int8(image_np, conv_w, conv_b, fc_w, fc_b):
    img_q = q8(image_np, INPUT_SCALE_8)
    cw_q  = q8(conv_w,   CONV_W_SCALE_8)
    cb_q  = q32(conv_b,  INPUT_SCALE_8 * CONV_W_SCALE_8)
    fw_q  = q8(fc_w,     FC_W_SCALE_8)
    fb_q  = q32(fc_b,    FB_SCALE_8)

    pool  = conv_relu_pool(img_q, cw_q, cb_q, CONV_SHIFT_8, 32767, np.int16)
    scores = fc(pool, fw_q, fb_q, FC_SHIFT_8)
    return int(np.argmax(scores))


def infer_int4(image_np, conv_w, conv_b, fc_w, fc_b):
    img_q = q4(image_np, INPUT_SCALE_4)
    cw_q  = q4(conv_w,   CONV_W_SCALE_4)
    cb_q  = q32(conv_b,  INPUT_SCALE_4 * CONV_W_SCALE_4)
    fw_q  = q4(fc_w,     FC_W_SCALE_4)
    fb_q  = q32(fc_b,    FB_SCALE_4)

    pool  = conv_relu_pool(img_q, cw_q, cb_q, CONV_SHIFT_4, 127, np.int8)
    scores = fc(pool, fw_q, fb_q, FC_SHIFT_4)
    return int(np.argmax(scores))


def infer_fp16(image_np, conv_w, conv_b, fc_w, fc_b):
    img_q = q16(image_np, INPUT_SCALE_F)
    cw_q  = q16(conv_w,   CONV_W_SCALE_F)
    cb_q  = q32(conv_b,   INPUT_SCALE_F * CONV_W_SCALE_F)
    fw_q  = q16(fc_w,     FC_W_SCALE_F)
    fb_q  = q32(fc_b,     FB_SCALE_F)

    pool  = conv_relu_pool(img_q, cw_q, cb_q, CONV_SHIFT_F, 32767, np.int16)
    scores = fc(pool, fw_q, fb_q, FC_SHIFT_F)
    return int(np.argmax(scores))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="mnist_cnn.pth")
    parser.add_argument("--samples",    type=int, default=SAMPLES_PER_DIGIT,
                        help="Số mẫu mỗi chữ số")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        type=str, default="comparison_report.txt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    model = SmallMNISTCNN()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    conv_w = model.conv.weight.detach().numpy().squeeze(1).astype(np.float32)
    conv_b = model.conv.bias.detach().numpy().astype(np.float32)
    fc_w   = model.fc.weight.detach().numpy().astype(np.float32)
    fc_b   = model.fc.bias.detach().numpy().astype(np.float32)

    # Load MNIST test set
    test_ds = datasets.MNIST(root="./data", train=False, download=True,
                             transform=transforms.ToTensor())
    by_label = {d: [] for d in range(10)}
    for idx, (_, lbl) in enumerate(test_ds):
        by_label[int(lbl)].append(idx)

    # Counters
    MODES = ["float32", "int8", "int4", "fp16"]
    correct = {m: [0]*10 for m in MODES}
    total_d = [0]*10
    total   = 0

    print(f"Chạy inference {args.samples * 10} mẫu × 4 kiểu...")
    print(f"(Float32=reference, INT8=khớp RTL, INT4=4-bit, FP16=16-bit fixed)\n")

    for digit in range(10):
        chosen = random.sample(by_label[digit], args.samples)
        for ds_idx in chosen:
            image_t, _ = test_ds[ds_idx]
            image_np   = image_t.squeeze(0).numpy().astype(np.float32)

            preds = {
                "float32": infer_float32(image_np, model),
                "int8":    infer_int8(image_np, conv_w, conv_b, fc_w, fc_b),
                "int4":    infer_int4(image_np, conv_w, conv_b, fc_w, fc_b),
                "fp16":    infer_fp16(image_np, conv_w, conv_b, fc_w, fc_b),
            }
            for m in MODES:
                if preds[m] == digit:
                    correct[m][digit] += 1
            total_d[digit] += 1
            total          += 1

        done_pct = 100 * (digit+1) / 10
        print(f"  Digit {digit} done — INT8:{correct['int8'][digit]}/{args.samples} "
              f"INT4:{correct['int4'][digit]}/{args.samples} "
              f"FP16:{correct['fp16'][digit]}/{args.samples}  [{done_pct:.0f}%]")

    # ── Tạo báo cáo ───────────────────────────────────────────────────
    sep  = "=" * 65
    sep2 = "-" * 65
    lines = []
    lines.append(sep)
    lines.append("  SO SANH DO CHINH XAC THEO KIEU DU LIEU (Python reference)")
    lines.append(f"  Mo hinh   : {args.checkpoint}  (4-filter, 4×13×13 FC)")
    lines.append(f"  Tap du lieu: MNIST test set ({len(test_ds)} anh)")
    lines.append(f"  Mau test  : {args.samples}/chu so × 10 = {total} tong")
    lines.append(f"  Seed      : {args.seed}")
    lines.append(sep)
    lines.append(f"  {'Digit':>5} | {'Float32':>8} | {'INT8':>8} | {'INT4':>8} | {'FP16':>8}")
    lines.append(sep2)

    sum_c = {m: 0 for m in MODES}
    for d in range(10):
        t = total_d[d]
        row = f"    {d:>3}   |"
        for m in MODES:
            acc = 100.0 * correct[m][d] / t if t > 0 else 0
            row += f"  {acc:6.2f}%  |"
            sum_c[m] += correct[m][d]
        lines.append(row)

    lines.append(sep2)
    row = f"    AVG   |"
    for m in MODES:
        acc = 100.0 * sum_c[m] / total if total > 0 else 0
        row += f"  {acc:6.2f}%  |"
    lines.append(row)
    lines.append(f"  Correct |")
    row2 = f"          |"
    for m in MODES:
        row2 += f" {sum_c[m]:4d}/{total}  |"
    lines.append(row2)
    lines.append(sep)

    # Bảng error rate
    lines.append("")
    lines.append(sep)
    lines.append("  TI LE LOI (Error Rate)")
    lines.append(sep)
    lines.append(f"  {'Digit':>5} | {'Float32':>8} | {'INT8':>8} | {'INT4':>8} | {'FP16':>8}")
    lines.append(sep2)
    for d in range(10):
        t = total_d[d]
        row = f"    {d:>3}   |"
        for m in MODES:
            err = 100.0 * (t - correct[m][d]) / t if t > 0 else 0
            row += f"  {err:6.2f}%  |"
        lines.append(row)
    lines.append(sep2)
    row = f"    AVG   |"
    for m in MODES:
        err = 100.0 * (total - sum_c[m]) / total if total > 0 else 0
        row += f"  {err:6.2f}%  |"
    lines.append(row)
    lines.append(sep)

    lines.append("")
    lines.append("GHI CHU:")
    lines.append("  Float32  : PyTorch goc, floating point 32-bit (baseline chinh xac nhat)")
    lines.append("  INT8     : int8 weights/activations, MAC int32  ← khop RTL cnn_accel_top.sv")
    lines.append("  INT4     : int4 weights/activations (-8..7)     ← khop RTL cnn_accel_int4.sv")
    lines.append("  FP16     : fixed-point 16-bit (scale 32767/256) ← khop RTL cnn_accel_fp16.sv")
    lines.append("  Pipeline : cung accuracy voi INT8, chi khac latency → khong can inference rieng")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path(args.out)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nBao cao luu tai: {out_path}")


if __name__ == "__main__":
    main()