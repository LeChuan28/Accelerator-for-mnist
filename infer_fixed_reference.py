import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from train_mnist import SmallMNISTCNN

INPUT_SCALE = 127
CONV_W_SCALE = 64
CONV_OUT_SCALE = 64
FC_W_SCALE = 32
CONV_SHIFT = 7
FC_SHIFT = 6


def quantize_to_int8(arr: np.ndarray, scale: float) -> np.ndarray:
    q = np.round(arr * scale)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)


def quantize_bias_to_int32(arr: np.ndarray, scale: float) -> np.ndarray:
    q = np.round(arr * scale)
    return q.astype(np.int32)


def conv2d_valid_single_channel(image_q: np.ndarray, weight_q: np.ndarray, bias_q: np.ndarray) -> np.ndarray:
    # image_q: (28,28) int8
    # weight_q: (4,3,3) int8
    out = np.zeros((4, 26, 26), dtype=np.int32)
    for oc in range(4):
        for y in range(26):
            for x in range(26):
                acc = int(bias_q[oc])
                for ky in range(3):
                    for kx in range(3):
                        acc += int(image_q[y + ky, x + kx]) * int(weight_q[oc, ky, kx])
                acc = acc >> CONV_SHIFT
                acc = max(acc, 0)
                acc = min(acc, 32767)
                out[oc, y, x] = acc
    return out.astype(np.int16)


def maxpool2x2(feat: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 13, 13), dtype=np.int16)
    for oc in range(4):
        for y in range(13):
            for x in range(13):
                block = feat[oc, y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]
                out[oc, y, x] = np.max(block)
    return out


def fc_layer(pool_feat: np.ndarray, fc_w_q: np.ndarray, fc_b_q: np.ndarray) -> np.ndarray:
    flat = pool_feat.reshape(-1).astype(np.int32)
    scores = np.zeros((10,), dtype=np.int32)
    for c in range(10):
        acc = int(fc_b_q[c])
        for i in range(flat.shape[0]):
            acc += int(flat[i]) * int(fc_w_q[c, i])
        scores[c] = acc >> FC_SHIFT
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    model = SmallMNISTCNN()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    image_t, label = test_ds[args.sample_idx]
    image_np = image_t.squeeze(0).numpy().astype(np.float32)

    conv_w = model.conv.weight.detach().numpy().squeeze(1).astype(np.float32)  # (4,3,3)
    conv_b = model.conv.bias.detach().numpy().astype(np.float32)
    fc_w = model.fc.weight.detach().numpy().astype(np.float32)  # (10,676)
    fc_b = model.fc.bias.detach().numpy().astype(np.float32)

    image_q = quantize_to_int8(image_np, INPUT_SCALE)
    conv_w_q = quantize_to_int8(conv_w, CONV_W_SCALE)
    conv_b_q = quantize_bias_to_int32(conv_b, INPUT_SCALE * CONV_W_SCALE)
    fc_w_q = quantize_to_int8(fc_w, FC_W_SCALE)
    fc_b_q = quantize_bias_to_int32(fc_b, CONV_OUT_SCALE * FC_W_SCALE)

    conv_out = conv2d_valid_single_channel(image_q, conv_w_q, conv_b_q)
    pool_out = maxpool2x2(conv_out)
    scores = fc_layer(pool_out, fc_w_q, fc_b_q)
    pred = int(np.argmax(scores))

    with torch.no_grad():
        float_pred = int(torch.argmax(model(image_t.unsqueeze(0)), dim=1).item())

    print(f"Label        : {label}")
    print(f"Float pred   : {float_pred}")
    print(f"Fixed pred   : {pred}")
    print(f"Fixed scores : {scores.tolist()}")


if __name__ == "__main__":
    main()
