"""Print the structure of a checkpoint.

Usage: python scripts/dev/check_ckpt.py [checkpoint.pt]
Defaults to the shipped checkpoints/loco_stage2.pt.
"""
import os
import sys

import torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "loco_stage2.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("Top-level keys:", list(ckpt.keys()))
for key in ckpt:
    val = ckpt[key]
    if isinstance(val, dict):
        print(f"\n  {key} (dict, {len(val)} keys):")
        for k2 in sorted(val.keys()):
            v2 = val[k2]
            if hasattr(v2, 'shape'):
                print(f"    {k2}: {v2.shape}")
            else:
                print(f"    {k2}: {v2}")
    elif hasattr(val, 'shape'):
        print(f"  {key}: {val.shape}")
    else:
        print(f"  {key}: {val}")
