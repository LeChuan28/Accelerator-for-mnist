import argparse
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
    CONV_SHIFT,
    FC_SHIFT,
    quantize_to_int8,
    quantize_bias_to_int32,
    conv2d_valid_single_channel,
    maxpool2x2,
    fc_layer,
)


def to_hex_twos(value: int, bits: int) -> str:
    mask = (1 << bits) - 1
    return f"{(value & mask):0{bits // 4}X}"


def write_hex_file(path: Path, values: np.ndarray, bits: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        flat = values.reshape(-1)
        for v in flat:
            f.write(to_hex_twos(int(v), bits) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model = SmallMNISTCNN()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    conv_w = model.conv.weight.detach().numpy().squeeze(1).astype(np.float32)
    conv_b = model.conv.bias.detach().numpy().astype(np.float32)
    fc_w = model.fc.weight.detach().numpy().astype(np.float32)
    fc_b = model.fc.bias.detach().numpy().astype(np.float32)

    conv_w_q = quantize_to_int8(conv_w, CONV_W_SCALE)
    conv_b_q = quantize_bias_to_int32(conv_b, INPUT_SCALE * CONV_W_SCALE)
    fc_w_q = quantize_to_int8(fc_w, FC_W_SCALE)
    fc_b_q = quantize_bias_to_int32(fc_b, CONV_OUT_SCALE * FC_W_SCALE)

    write_hex_file(export_dir / "conv_w.hex", conv_w_q, 8)
    write_hex_file(export_dir / "conv_b.hex", conv_b_q, 32)
    write_hex_file(export_dir / "fc_w.hex", fc_w_q, 8)
    write_hex_file(export_dir / "fc_b.hex", fc_b_q, 32)

    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    labels_lines = []
    pred_lines = []
    for idx in range(args.num_samples):
        image_t, label = test_ds[idx]
        image_np = image_t.squeeze(0).numpy().astype(np.float32)
        image_q = quantize_to_int8(image_np, INPUT_SCALE)
        write_hex_file(export_dir / f"image_{idx}.hex", image_q, 8)
        labels_lines.append(f"image_{idx}.hex label={label}")

        conv_out = conv2d_valid_single_channel(image_q, conv_w_q, conv_b_q)
        pool_out = maxpool2x2(conv_out)
        scores = fc_layer(pool_out, fc_w_q, fc_b_q)
        pred = int(np.argmax(scores))
        pred_lines.append(f"image_{idx}.hex pred={pred} scores={scores.tolist()}")

    (export_dir / "sample_labels.txt").write_text("\n".join(labels_lines) + "\n", encoding="utf-8")
    (export_dir / "reference_predictions.txt").write_text("\n".join(pred_lines) + "\n", encoding="utf-8")
    (export_dir / "quant_params.txt").write_text(
        "\n".join(
            [
                f"INPUT_SCALE={INPUT_SCALE}",
                f"CONV_W_SCALE={CONV_W_SCALE}",
                f"CONV_OUT_SCALE={CONV_OUT_SCALE}",
                f"FC_W_SCALE={FC_W_SCALE}",
                f"CONV_SHIFT={CONV_SHIFT}",
                f"FC_SHIFT={FC_SHIFT}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Export complete to: {export_dir}")


if __name__ == "__main__":
    main()
