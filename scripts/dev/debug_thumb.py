#!/usr/bin/env python3
r"""
Quick diagnostic: dump DEX3 thumb joint limits and test close directions.
Run from C:\IsaacLab:
  .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\high_low_hierarchical_g1\scripts\dev\debug_thumb.py --headless
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys, os, torch
sys.stdout.reconfigure(line_buffering=True)

import isaaclab.sim as sim_utils

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG_PARENT = os.path.dirname(_PKG_DIR)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from high_low_hierarchical_g1.envs.hierarchical_env import (
    HierarchicalG1Env, HierarchicalSceneCfg, PHYSICS_DT,
)

device = "cuda:0"
sim_cfg = sim_utils.SimulationCfg(
    dt=PHYSICS_DT, device=device, gravity=(0.0, 0.0, -9.81),
    physx=sim_utils.PhysxCfg(solver_type=1),
)
sim = sim_utils.SimulationContext(sim_cfg)

# Dummy loco checkpoint (we just need scene setup, not actual policy)
# Shipped Stage 2 locomotion checkpoint
LOCO_CKPT = os.path.join(_PKG_DIR, "checkpoints", "loco_stage2.pt")

scene_cfg = HierarchicalSceneCfg()
env = HierarchicalG1Env(
    sim=sim, scene_cfg=scene_cfg,
    checkpoint_path=LOCO_CKPT,
    num_envs=1, device=device,
)
obs = env.reset()

# Get joint limits from PhysX
joint_limits = env.robot.root_physx_view.get_dof_limits()  # [N, num_joints, 2]
joint_names = env.robot.joint_names

print("\n" + "=" * 70)
print("  DEX3 FINGER JOINT LIMITS")
print("=" * 70)

for i, name in enumerate(joint_names):
    if "hand" in name:
        lo = joint_limits[0, i, 0].item()
        hi = joint_limits[0, i, 1].item()
        cur = env.robot.data.joint_pos[0, i].item()
        print(f"  {name:40s} | range: [{lo:+.3f}, {hi:+.3f}] | current: {cur:+.3f}")

print("\n" + "=" * 70)
print("  TESTING THUMB DIRECTIONS")
print("=" * 70)

# Find thumb joint indices
thumb_joints = {}
for i, name in enumerate(joint_names):
    if "thumb" in name and "right" in name:
        thumb_joints[name] = i

# Test: set each thumb joint to +0.5 and -0.5, step, and see effect
for name, idx in thumb_joints.items():
    for val in [+0.6, -0.6]:
        # Reset to zero
        targets = env.robot.data.joint_pos.clone()
        targets[0, idx] = val
        env.robot.set_joint_position_target(targets)
        env.scene.write_data_to_sim()
        # Step a few times
        for _ in range(20):
            sim.step()
            env.scene.update(PHYSICS_DT)
        actual = env.robot.data.joint_pos[0, idx].item()
        # Get body positions for palm
        palm_pos = env.robot.data.body_pos_w[0, 40]  # right_hand_palm_link
        print(f"  {name}: target={val:+.2f} -> actual={actual:+.3f}")

# Reset back
obs = env.reset()

# Try closing all fingers with positive values and observe
print("\n" + "=" * 70)
print("  CLOSE TEST: ALL +0.6 rad")
print("=" * 70)

targets = env.robot.data.joint_pos.clone()
for name, idx in thumb_joints.items():
    targets[0, idx] = 0.6
    print(f"  Setting {name} = +0.6")

# Also set index/middle
for i, name in enumerate(joint_names):
    if "right_hand_index" in name or "right_hand_middle" in name:
        targets[0, i] = 0.8

env.robot.set_joint_position_target(targets)
for _ in range(50):
    env.scene.write_data_to_sim()
    sim.step()
    env.scene.update(PHYSICS_DT)

print("  After 50 steps with positive targets:")
for i, name in enumerate(joint_names):
    if "hand" in name and "right" in name:
        actual = env.robot.data.joint_pos[0, i].item()
        print(f"    {name:40s} = {actual:+.3f}")

# Now try negative thumb_0
print("\n" + "=" * 70)
print("  CLOSE TEST: thumb_0 = -0.6, others +0.6")
print("=" * 70)

obs = env.reset()
targets = env.robot.data.joint_pos.clone()
for name, idx in thumb_joints.items():
    if "thumb_0" in name:
        targets[0, idx] = -0.6
        print(f"  Setting {name} = -0.6")
    else:
        targets[0, idx] = 0.6
        print(f"  Setting {name} = +0.6")

for i, name in enumerate(joint_names):
    if "right_hand_index" in name or "right_hand_middle" in name:
        targets[0, i] = 0.8

env.robot.set_joint_position_target(targets)
for _ in range(50):
    env.scene.write_data_to_sim()
    sim.step()
    env.scene.update(PHYSICS_DT)

print("  After 50 steps with negative thumb_0:")
for i, name in enumerate(joint_names):
    if "hand" in name and "right" in name:
        actual = env.robot.data.joint_pos[0, i].item()
        print(f"    {name:40s} = {actual:+.3f}")

print("\n[Done]")
env.close()
simulation_app.close()
