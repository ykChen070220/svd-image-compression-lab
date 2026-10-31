"""Restore a saved factor archive: python restore.py svd_k30.npz output.png"""
import argparse
import numpy as np
from core import display_pixels, png_bytes

def main():
    parser = argparse.ArgumentParser(description="将实验室导出的 SVD 因子还原为 PNG")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        shape = tuple(data["shape"])
        channels = 1 if len(shape) == 2 else shape[2]
        parts = [(data[f"U_{c}"] * data[f"s_{c}"]) @ data[f"Vh_{c}"] for c in range(channels)]
    image = parts[0] if len(shape) == 2 else np.stack(parts, axis=-1)
    with open(args.output, "wb") as target:
        target.write(png_bytes(display_pixels(image)))

if __name__ == "__main__":
    main()
