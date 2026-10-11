"""
evaluate_mnist.py — Đánh giá toàn diện SmallMNISTCNN
=====================================================
Chạy:
    python evaluate_mnist.py --checkpoint mnist_cnn.pth

Tự động tìm file <checkpoint>.history.json (do train_mnist.py tạo ra)
để vẽ thêm đồ thị Loss và Accuracy theo epoch.

Kết quả xuất ra:
    - In terminal            : accuracy, per-class metrics, confusion matrix text
    - <prefix>_loss.png      : biểu đồ Train/Test Loss theo epoch
    - <prefix>_accuracy.png  : biểu đồ Train/Test Accuracy theo epoch
    - <prefix>_confusion_matrix.png : Confusion Matrix (Float, % theo hàng)
      (prefix mặc định: "evaluation_report", đổi bằng --out-prefix)

Yêu cầu:
    pip install torch torchvision numpy matplotlib seaborn scikit-learn
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

# ── Import từ các file có sẵn ─────────────────────────────────────────────────
from train_mnist import SmallMNISTCNN
from infer_fixed_reference import (
    INPUT_SCALE,
    CONV_W_SCALE,
    CONV_OUT_SCALE,
    FC_W_SCALE,
    quantize_to_int8,
    quantize_bias_to_int32,
    conv2d_valid_single_channel,
    maxpool2x2,
    fc_layer,
)

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib không tìm thấy — bỏ qua vẽ biểu đồ.\n"
          "       pip install matplotlib seaborn scikit-learn")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

try:
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
    HAS_SKL = True
except ImportError:
    HAS_SKL = False
    print("[WARN] scikit-learn không tìm thấy — một số metrics sẽ bị bỏ qua.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Thu thập dự đoán
# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(model, test_ds, device):
    model.eval()

    conv_w = model.conv.weight.detach().numpy().squeeze(1).astype(np.float32)
    conv_b = model.conv.bias.detach().numpy().astype(np.float32)
    fc_w   = model.fc.weight.detach().numpy().astype(np.float32)
    fc_b   = model.fc.bias.detach().numpy().astype(np.float32)

    conv_w_q = quantize_to_int8(conv_w, CONV_W_SCALE)
    conv_b_q = quantize_bias_to_int32(conv_b, INPUT_SCALE * CONV_W_SCALE)
    fc_w_q   = quantize_to_int8(fc_w, FC_W_SCALE)
    fc_b_q   = quantize_bias_to_int32(fc_b, CONV_OUT_SCALE * FC_W_SCALE)

    labels, float_preds, fixed_preds, float_logits_all = [], [], [], []

    print(f"Đang chạy inference trên {len(test_ds)} mẫu test...")
    for idx, (image_t, label) in enumerate(test_ds):
        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(test_ds)}")

        with torch.no_grad():
            logits = model(image_t.unsqueeze(0).to(device)).cpu().squeeze(0)
        float_pred = int(torch.argmax(logits).item())
        float_logits_all.append(logits.numpy())

        image_np   = image_t.squeeze(0).numpy().astype(np.float32)
        image_q    = quantize_to_int8(image_np, INPUT_SCALE)
        conv_out   = conv2d_valid_single_channel(image_q, conv_w_q, conv_b_q)
        pool_out   = maxpool2x2(conv_out)
        scores     = fc_layer(pool_out, fc_w_q, fc_b_q)
        fixed_pred = int(np.argmax(scores))

        labels.append(label)
        float_preds.append(float_pred)
        fixed_preds.append(fixed_pred)

    labels      = np.array(labels)
    float_preds = np.array(float_preds)
    fixed_preds = np.array(fixed_preds)
    float_probs = torch.softmax(torch.tensor(np.array(float_logits_all)), dim=1).numpy()
    return labels, float_preds, fixed_preds, float_probs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tính metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(labels, preds, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm):
    rows = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
        rows.append((c, precision, recall, f1, int(cm[c, :].sum())))
    return rows


def overfitting_indicators(labels, float_preds, float_probs):
    max_conf = float_probs.max(axis=1)
    wrong    = (float_preds != labels)
    high_conf_wrong = ((max_conf > 0.9) & wrong).sum()
    pred_counts  = np.bincount(float_preds, minlength=10)
    label_counts = np.bincount(labels, minlength=10)
    bias_ratio   = pred_counts.max() / max(pred_counts.min(), 1)
    return {
        "mean_confidence"      : float(max_conf.mean()),
        "median_confidence"    : float(np.median(max_conf)),
        "high_conf_wrong_pct"  : float(high_conf_wrong / len(labels) * 100),
        "pred_class_bias_ratio": float(bias_ratio),
        "pred_counts"          : pred_counts,
        "label_counts"         : label_counts,
    }


def print_report(labels, float_preds, fixed_preds, float_probs):
    total     = len(labels)
    float_acc = (float_preds == labels).mean() * 100
    fixed_acc = (fixed_preds == labels).mean() * 100
    agree_pct = (float_preds == fixed_preds).mean() * 100

    print("\n" + "═"*60)
    print("  KẾT QUẢ ĐÁNH GIÁ SmallMNISTCNN")
    print("═"*60)
    print(f"  Tổng mẫu test            : {total:,}")
    print(f"  Float accuracy           : {float_acc:.2f}%")
    print(f"  Fixed-point accuracy     : {fixed_acc:.2f}%")
    print(f"  Float vs Fixed đồng thuận: {agree_pct:.2f}%")

    cm = compute_confusion_matrix(labels, float_preds)
    print("\n── CONFUSION MATRIX (Float) ──────────────────────────────────")
    header = "      " + "  ".join(f"{i:4d}" for i in range(10))
    print(header)
    print("  " + "─"*(len(header)-2))
    for i in range(10):
        print(f"  {i} |  " + "  ".join(f"{cm[i,j]:4d}" for j in range(10)))

    print("\n── PER-CLASS METRICS (Float) ─────────────────────────────────")
    print(f"  {'Digit':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print("  " + "─"*54)
    metrics = per_class_metrics(cm)
    for c, prec, rec, f1, sup in metrics:
        flag = " ⚠" if f1 < 0.90 else ""
        print(f"  {c:>5}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}  {sup:>8}{flag}")
    avg_p = np.mean([m[1] for m in metrics])
    avg_r = np.mean([m[2] for m in metrics])
    avg_f = np.mean([m[3] for m in metrics])
    print("  " + "─"*54)
    print(f"  {'macro':>5}  {avg_p:>10.4f}  {avg_r:>8.4f}  {avg_f:>8.4f}")

    ovi = overfitting_indicators(labels, float_preds, float_probs)
    print("\n── KIỂM TRA HỌC VẸT / OVERFITTING ──────────────────────────")
    print(f"  Confidence trung bình        : {ovi['mean_confidence']:.4f}")
    print(f"  Confidence trung vị          : {ovi['median_confidence']:.4f}")
    print(f"  Sai nhưng confidence > 90%   : {ovi['high_conf_wrong_pct']:.2f}%  "
          f"({'⚠ ĐÁNG CHÚ Ý' if ovi['high_conf_wrong_pct'] > 2 else 'OK'})")
    print(f"  Bias tỉ lệ dự đoán giữa class: {ovi['pred_class_bias_ratio']:.2f}x  "
          f"({'⚠ MẤT CÂN BẰNG' if ovi['pred_class_bias_ratio'] > 5 else 'OK'})")

    print(f"\n  {'Digit':>5}  {'Actual':>8}  {'Predicted':>10}  {'Δ':>6}")
    for i in range(10):
        delta = int(ovi['pred_counts'][i]) - int(ovi['label_counts'][i])
        flag  = " ⚠" if abs(delta) > 200 else ""
        print(f"  {i:>5}  {ovi['label_counts'][i]:>8}  {ovi['pred_counts'][i]:>10}  {delta:>+6}{flag}")

    if HAS_SKL:
        kappa = cohen_kappa_score(labels, float_preds)
        mcc   = matthews_corrcoef(labels, float_preds)
        print(f"\n  Cohen's Kappa : {kappa:.4f}  (>0.8 = tốt, >0.9 = rất tốt)")
        print(f"  Matthews MCC  : {mcc:.4f}  (>0.9 = mô hình tin cậy)")

    print("\n── NHẬN XÉT ─────────────────────────────────────────────────")
    verdicts = []
    if float_acc >= 98:
        verdicts.append("✅ Accuracy ≥ 98%: Model học tốt, tổng quát hoá cao.")
    elif float_acc >= 95:
        verdicts.append("🟡 Accuracy 95–98%: Khá tốt, có thể cải thiện thêm.")
    else:
        verdicts.append("🔴 Accuracy < 95%: Cần xem lại kiến trúc hoặc dữ liệu.")
    if ovi['high_conf_wrong_pct'] > 5:
        verdicts.append("⚠️  Model hay sai nhưng vẫn rất tự tin → dấu hiệu overfit.")
    else:
        verdicts.append("✅ Tỉ lệ sai-với-confidence-cao thấp → model biết mình không chắc.")
    if ovi['pred_class_bias_ratio'] > 5:
        verdicts.append("⚠️  Dự đoán thiên lệch class → có thể đang thiên vị nhãn phổ biến.")
    else:
        verdicts.append("✅ Phân phối dự đoán cân bằng → không thiên vị class cụ thể.")
    if agree_pct >= 98:
        verdicts.append("✅ Float và Fixed-point nhất quán cao → lượng tử hóa an toàn.")
    else:
        verdicts.append(f"⚠️  Float vs Fixed chỉ đồng thuận {agree_pct:.1f}% → cần kiểm tra scale.")
    for v in verdicts:
        print(f"  {v}")
    print("═"*60 + "\n")
    return cm, metrics, ovi


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vẽ biểu đồ
# ─────────────────────────────────────────────────────────────────────────────

BG      = "#0f172a"   # nền tổng
PANEL   = "#1e293b"   # nền từng panel
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
BORDER  = "#334155"
GREEN   = "#22c55e"
AMBER   = "#f59e0b"
RED     = "#ef4444"
BLUE    = "#3b82f6"
PURPLE  = "#a855f7"
TEAL    = "#2dd4bf"


def _style(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


# Các epoch cần ghi rõ giá trị ngay trên biểu đồ (đối chiếu nhanh cho báo cáo)
MARK_EPOCHS = (1, 5, 10, 15, 20)


def _mark_epoch_annotations(ax, epochs, values, color, dy=8, fmt="{:.4f}"):
    """Ghi giá trị tại các epoch nằm trong MARK_EPOCHS (bỏ qua nếu epoch đó
    không tồn tại trong history, ví dụ history chỉ có 10 epoch thì bỏ qua 15/20)."""
    epoch_to_idx = {e: i for i, e in enumerate(epochs)}
    for me in MARK_EPOCHS:
        if me in epoch_to_idx:
            idx = epoch_to_idx[me]
            v = values[idx]
            ax.annotate(fmt.format(v), xy=(me, v), xytext=(0, dy),
                        textcoords="offset points", color=color, fontsize=7.5,
                        ha="center", fontweight="bold")


def plot_loss(history, out_path: Path):
    """Xuất RIÊNG 1 file PNG: Train/Test Loss theo epoch."""
    if not HAS_MPL:
        return
    if history is None or len(history.get("epoch", [])) == 0:
        print("[INFO] Không có history — bỏ qua vẽ biểu đồ Loss.")
        return

    epochs        = history["epoch"]
    has_test_loss = "test_loss" in history and len(history["test_loss"]) == len(epochs)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    _style(ax, "Loss theo epoch (Train/Test)" if has_test_loss else "Training Loss theo epoch")

    ax.plot(epochs, history["train_loss"],
            color=RED, linewidth=2, marker="o", markersize=5, label="Train loss")
    _mark_epoch_annotations(ax, epochs, history["train_loss"], RED, dy=8, fmt="{:.4f}")

    if has_test_loss:
        ax.plot(epochs, history["test_loss"],
                color=TEAL, linewidth=2, marker="s", markersize=5, label="Test loss")
        _mark_epoch_annotations(ax, epochs, history["test_loss"], TEAL, dy=-16, fmt="{:.4f}")
    else:
        print("[INFO] history.json không có khóa 'test_loss' — chỉ vẽ Train loss.\n"
              "       Muốn có cả Test loss, hãy ghi thêm 'test_loss' mỗi epoch trong train_mnist.py.")

    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax.set_ylabel("Loss (CrossEntropy)", color=MUTED, fontsize=9)
    ax.set_xticks(epochs)
    ax.legend(fontsize=9, labelcolor=TEXT, facecolor=PANEL, edgecolor=BORDER)
    ax.tick_params(axis="y", colors=MUTED)
    ax.tick_params(axis="x", colors=MUTED)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Đã lưu biểu đồ Loss: {out_path}")


def plot_accuracy(history, out_path: Path):
    """Xuất RIÊNG 1 file PNG: Train/Test Accuracy theo epoch."""
    if not HAS_MPL:
        return
    if history is None or len(history.get("epoch", [])) == 0:
        print("[INFO] Không có history — bỏ qua vẽ biểu đồ Accuracy.")
        return

    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    _style(ax, "Accuracy theo epoch (Train/Test)")

    ax.plot(epochs, [v * 100 for v in history["train_acc"]],
            color=TEAL, linewidth=2, marker="o", markersize=5,
            label="Train acc", linestyle="--")
    ax.plot(epochs, [v * 100 for v in history["test_acc"]],
            color=GREEN, linewidth=2, marker="s", markersize=5,
            label="Test acc")
    _mark_epoch_annotations(ax, epochs, [v * 100 for v in history["train_acc"]],
                             TEAL, dy=10, fmt="{:.2f}%")
    _mark_epoch_annotations(ax, epochs, [v * 100 for v in history["test_acc"]],
                             GREEN, dy=-18, fmt="{:.2f}%")
    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax.set_ylabel("Accuracy (%)", color=MUTED, fontsize=9)
    ax.set_xticks(epochs)
    ax.set_ylim(
        max(0, min(min(history["train_acc"]), min(history["test_acc"])) * 100 - 2),
        101
    )
    ax.legend(fontsize=9, labelcolor=TEXT, facecolor=PANEL, edgecolor=BORDER)

    # Gap train vs test (chỉ báo overfit)
    # Đặt cố định ở góc dưới-phải trong khung trục (axes fraction), KHÔNG neo vào
    # điểm epoch cuối nữa để tránh đè lên nhãn % của _mark_epoch_annotations.
    gap = (history["train_acc"][-1] - history["test_acc"][-1]) * 100
    gap_color = AMBER if gap > 1.0 else GREEN
    ax.text(0.99, 0.04, f"Gap (train-test): {gap:+.2f}%",
            transform=ax.transAxes,
            ha="right", va="bottom",
            color=gap_color, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                      edgecolor=gap_color, linewidth=0.8))
    ax.tick_params(axis="y", colors=MUTED)
    ax.tick_params(axis="x", colors=MUTED)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Đã lưu biểu đồ Accuracy: {out_path}")


def plot_confusion_matrix(cm, out_path: Path):
    """Xuất RIÊNG 1 file PNG: Confusion Matrix (Float, % theo hàng)."""
    if not HAS_MPL:
        return

    cmap = LinearSegmentedColormap.from_list("mnist_cm", ["#f8fafc", "#1e40af"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)

    if HAS_SNS:
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap=cmap,
                    ax=ax, linewidths=0.5, linecolor=BG,
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 9, "color": "#1e293b"})
    else:
        im = ax.imshow(cm_norm, cmap=cmap, aspect="auto")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8,
                        color="white" if cm_norm[i, j] < 50 else BG)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_facecolor(PANEL)
    ax.set_xlabel("Predicted", color=MUTED, fontsize=10)
    ax.set_ylabel("Actual",    color=MUTED, fontsize=10)
    ax.set_title("Confusion Matrix (Float, % theo hàng)", color=TEXT,
                 fontsize=12, pad=10, fontweight="bold")
    ax.tick_params(colors=MUTED)
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors=MUTED, labelsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Đã lưu biểu đồ Confusion Matrix: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá toàn diện SmallMNISTCNN (float + fixed-point)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Đường dẫn tới file .pth đã train")
    parser.add_argument("--out-prefix", type=str, default="evaluation_report",
                        help="Tiền tố tên file ảnh xuất ra (mặc định: evaluation_report). "
                             "Sẽ tạo 3 file: <prefix>_loss.png, <prefix>_accuracy.png, "
                             "<prefix>_confusion_matrix.png")
    parser.add_argument("--no-plot", action="store_true",
                        help="Bỏ qua vẽ biểu đồ")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SmallMNISTCNN()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    test_ds = datasets.MNIST(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor()
    )

    labels, float_preds, fixed_preds, float_probs = collect_predictions(
        model, test_ds, device
    )
    cm, metrics, ovi = print_report(labels, float_preds, fixed_preds, float_probs)

    # ── Đọc history (nếu có) ──────────────────────────────────────────────────
    history      = None
    history_path = Path(args.checkpoint).with_suffix(".history.json")
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"[✓] Đọc history từ: {history_path}  ({len(history['epoch'])} epochs)")
    else:
        print(f"[INFO] Không tìm thấy {history_path} — bỏ qua đồ thị Loss/Accuracy.\n"
              f"       Train lại với train_mnist.py mới để tạo file history.")

    if not args.no_plot:
        prefix = args.out_prefix
        plot_loss(history, Path(f"{prefix}_loss.png"))
        plot_accuracy(history, Path(f"{prefix}_accuracy.png"))
        plot_confusion_matrix(cm, Path(f"{prefix}_confusion_matrix.png"))


if __name__ == "__main__":
    main()