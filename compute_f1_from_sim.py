"""
compute_f1_from_sim.py
Tính F1-Score, Precision, Recall từ kết quả simulation RTL.
Đọc từ sim_results/compare_raw.txt hoặc sim_results/accuracy_report.txt

Chạy từ thư mục gốc project:
    python python/compute_f1_from_sim.py --input sim_results/compare_raw.txt
    python python/compute_f1_from_sim.py --input sim_results/raw_transcript.txt
"""

import argparse, re
from pathlib import Path
from collections import defaultdict

def compute_metrics(true_labels, pred_labels, n_classes=10):
    """Tính Precision, Recall, F1 per class và macro average."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    results = {}
    for c in range(n_classes):
        p  = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        results[c] = {"precision": p, "recall": r, "f1": f1,
                      "tp": tp[c], "fp": fp[c], "fn": fn[c]}

    macro_p  = sum(r["precision"] for r in results.values()) / n_classes
    macro_r  = sum(r["recall"]    for r in results.values()) / n_classes
    macro_f1 = sum(r["f1"]        for r in results.values()) / n_classes
    accuracy = sum(tp.values()) / len(true_labels) if true_labels else 0

    return results, macro_p, macro_r, macro_f1, accuracy


def parse_compare_log(path):
    """Parse [CMP] lines từ tb_compare_all transcript."""
    data = {"int8": [], "int4": [], "fp16": [], "pip": [], "label": []}
    pattern = re.compile(
        r'\[CMP\].*label=(\d+)\s+int8=(\d+)\s+int4=(\d+)\s+fp16=(\d+)\s+pip=(\d+)')
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                lbl, p8, p4, pf, pp = [int(x) for x in m.groups()]
                data["label"].append(lbl)
                data["int8"].append(p8)
                data["int4"].append(p4)
                data["fp16"].append(pf)
                data["pip"].append(pp)
    return data


def parse_result_log(path):
    """Parse [RESULT] lines từ tb_cnn_accel_batch transcript."""
    labels, preds = [], []
    pattern = re.compile(r'\[RESULT\].*label=(\d+)\s+pred=(\d+)')
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                labels.append(int(m.group(1)))
                preds.append(int(m.group(2)))
    return labels, preds


def print_f1_table(results, macro_p, macro_r, macro_f1, accuracy, title=""):
    sep  = "=" * 72
    sep2 = "-" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"  Overall Accuracy: {accuracy*100:.2f}%  |  Macro-F1: {macro_f1*100:.2f}%")
    print(sep)
    print(f"  {'Digit':>5} | {'Precision':>10} | {'Recall':>8} | {'F1-Score':>9} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
    print(sep2)
    for c in range(10):
        r = results[c]
        print(f"    {c:>3}   | {r['precision']*100:9.2f}% | {r['recall']*100:7.2f}% | "
              f"{r['f1']*100:8.2f}% | {r['tp']:4d} | {r['fp']:4d} | {r['fn']:4d}")
    print(sep2)
    print(f"  {'MACRO':>5} | {macro_p*100:9.2f}% | {macro_r*100:7.2f}% | "
          f"{macro_f1*100:8.2f}% |      |      |")
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to transcript log file")
    parser.add_argument("--mode",   default="auto",
                        choices=["compare", "batch", "auto"],
                        help="Log format: compare=[CMP], batch=[RESULT]")
    parser.add_argument("--out",    default="sim_results/f1_report.txt")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        return

    # Detect mode
    mode = args.mode
    if mode == "auto":
        with open(path) as f:
            content = f.read(2000)
        mode = "compare" if "[CMP]" in content else "batch"
    print(f"Mode: {mode}")

    import sys
    out_lines = []

    if mode == "compare":
        data = parse_compare_log(path)
        n = len(data["label"])
        print(f"Parsed {n} samples")

        for variant, key in [("INT8","int8"), ("INT4","int4"),
                              ("FP16","fp16"), ("Pipeline","pip")]:
            if not data[key]: continue
            res, mp, mr, mf1, acc = compute_metrics(data["label"], data[key])
            print_f1_table(res, mp, mr, mf1, acc,
                           f"F1 REPORT — {variant} ({n} samples)")
            out_lines.append(f"\n{'='*72}")
            out_lines.append(f"  {variant} — Accuracy: {acc*100:.2f}%  Macro-F1: {mf1*100:.2f}%")
            out_lines.append(f"  {'Digit':>5} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
            out_lines.append(f"  {'-'*40}")
            for c in range(10):
                r = res[c]
                out_lines.append(f"    {c:>3}   | {r['precision']*100:6.2f}% | "
                                 f"{r['recall']*100:6.2f}% | {r['f1']*100:6.2f}%")
            out_lines.append(f"  MACRO | {mp*100:6.2f}% | {mr*100:6.2f}% | {mf1*100:6.2f}%")

    else:  # batch — single variant
        labels, preds = parse_result_log(path)
        n = len(labels)
        print(f"Parsed {n} samples")
        res, mp, mr, mf1, acc = compute_metrics(labels, preds)
        print_f1_table(res, mp, mr, mf1, acc, f"F1 REPORT — INT8 ({n} samples)")
        out_lines.append(f"Accuracy: {acc*100:.2f}%  Macro-F1: {mf1*100:.2f}%")
        for c in range(10):
            r = res[c]
            out_lines.append(f"  Digit {c}: P={r['precision']*100:.1f}%  "
                             f"R={r['recall']*100:.1f}%  F1={r['f1']*100:.1f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nF1 report saved: {out_path}")


if __name__ == "__main__":
    main()
