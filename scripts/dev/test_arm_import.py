"""Quick import test for the arm policy wrapper.

Usage: python scripts/dev/test_arm_import.py [checkpoint.pt]
Defaults to the shipped checkpoints/arm_stage2.pt.
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from high_low_hierarchical_g1.low_level.arm_policy_wrapper import (
    ArmPolicyWrapper, ARM_OBS_DIM, ARM_ACT_DIM,
    ARM_POLICY_JOINT_NAMES_29DOF, RIGHT_FINGER_JOINT_NAMES_29DOF,
    get_palm_forward, compute_orientation_error,
)

print(f"ArmPolicyWrapper imported: {ARM_OBS_DIM} obs -> {ARM_ACT_DIM} act")
print(f"Arm joints: {ARM_POLICY_JOINT_NAMES_29DOF}")
print(f"Right finger joints: {RIGHT_FINGER_JOINT_NAMES_29DOF}")

# Test network build
layers = []
prev = ARM_OBS_DIM
for h in [256, 256, 128]:
    layers += [nn.Linear(prev, h), nn.ELU()]
    prev = h
layers.append(nn.Linear(prev, ARM_ACT_DIM))
net = nn.Sequential(*layers)
test_in = torch.randn(4, ARM_OBS_DIM)
test_out = net(test_in)
print(f"Network test: {test_in.shape} -> {test_out.shape}")

# Test obs builder
obs = ArmPolicyWrapper.build_obs(
    arm_pos=torch.zeros(4, 7),
    arm_vel=torch.zeros(4, 7),
    ee_body=torch.zeros(4, 3),
    palm_quat=torch.tensor([[1, 0, 0, 0]] * 4, dtype=torch.float),
    target_body=torch.tensor([[0.3, -0.1, 0.1]] * 4),
    prev_arm_act=torch.zeros(4, 7),
    steps_since_spawn=torch.zeros(4, dtype=torch.long),
)
print(f"Obs builder test: {obs.shape} (expected [4, {ARM_OBS_DIM}])")

# Test checkpoint loading
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "arm_stage2.pt")
if os.path.exists(ckpt_path):
    wrapper = ArmPolicyWrapper(ckpt_path, device="cpu")
    action = wrapper.get_action(obs)
    targets = wrapper.get_arm_targets(obs)
    print(f"Inference: obs {obs.shape} -> action {action.shape} -> targets {targets.shape}")
    print(f"Sample action: {action[0].tolist()}")
    print(f"Sample targets: {targets[0].tolist()}")
else:
    sys.exit(f"FAILED: checkpoint not found: {ckpt_path}")

print("\nAll tests PASSED!")
