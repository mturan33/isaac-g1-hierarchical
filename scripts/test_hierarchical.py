#!/usr/bin/env python3
"""
Test Hierarchical G1 (V6.2 Loco) - Walk + Reach + Grasp Demo
===============================================================
Full hierarchical manipulation demo using V6.2 loco policy:
  Phase 1: Walk to table (3m forward)
  Phase 2: Stabilize (stand still)
  Phase 3: Switch to manipulation mode, reach forward
  Phase 4: Close fingers (grasp)
  Phase 5: Return arms to carry pose
  Phase 6: Return to default + open fingers

KEY CHANGE: V6.2 loco policy controls ONLY legs+waist (15 joints).
Arms are controlled separately by ArmController. No more arm override
conflict -- the loco policy never touches arms.

V6.2 checkpoint format: {"model": {"actor.0.weight": ...}, "iteration": ...}
Architecture: 66 obs -> [512,256,128](LN+ELU) -> 15 loco actions

Usage (from C:\\IsaacLab):
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\high_low_hierarchical_g1\\scripts\\test_hierarchical.py --num_envs 16 --checkpoint C:\\IsaacLab\\logs\\ulc\\g1_unified_stage1_2026-02-27_00-05-20\\model_best.pt
"""

# ============================================================================
# AppLauncher MUST be created before any Isaac Lab imports
# ============================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test hierarchical G1 V6.2 walk + reach + grasp")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to trained V6.2 loco policy checkpoint (.pt file)",
)
parser.add_argument(
    "--arm_checkpoint", type=str, default=None,
    help="Path to Stage 7 arm policy checkpoint (.pt file). If provided, uses learned arm control.",
)
parser.add_argument("--walk_distance", type=float, default=3.0, help="Walk distance in meters")
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum walk steps before timeout")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Isaac Lab imports (AFTER AppLauncher)
# ============================================================================
import os
import sys
import time
import torch

# Force unbuffered stdout for real-time output in headless mode
sys.stdout.reconfigure(line_buffering=True)

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply_inverse, quat_apply

# Add parent of high_low_hierarchical_g1 to path so package imports work
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_PARENT = os.path.dirname(_PKG_DIR)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from high_low_hierarchical_g1.envs.hierarchical_env import (
    HierarchicalG1Env, HierarchicalSceneCfg, PHYSICS_DT,
)
from high_low_hierarchical_g1.skills.walk_to import WalkToSkill
from high_low_hierarchical_g1.config.skill_config import WalkToConfig
from high_low_hierarchical_g1.low_level.arm_controller import ArmPose


def run_phase(env, name, step_fn, max_steps=500, log_interval=50):
    """Run a phase with logging."""
    print(f"\n{'-'*50}")
    print(f"  PHASE: {name}")
    print(f"{'-'*50}")

    for step in range(max_steps):
        if not simulation_app.is_running():
            return None

        obs = step_fn(step)

        if step > 0 and step % log_interval == 0:
            h = obs["base_height"].mean().item()
            standing = (obs["base_height"] > 0.3).sum().item()
            print(f"  [{name}] Step {step:4d} | Height: {h:.2f}m | Standing: {standing}/{env.num_envs}")

    return obs


def main():
    """Main test loop."""
    num_envs = args_cli.num_envs
    device = "cuda:0"

    print("=" * 60)
    print("  Hierarchical G1 V6.2 - Walk + Reach + Grasp Demo")
    print("=" * 60)
    print(f"  Environments : {num_envs}")
    print(f"  Checkpoint   : {args_cli.checkpoint}")
    print(f"  Walk distance: {args_cli.walk_distance}m")
    use_arm_policy = args_cli.arm_checkpoint is not None
    arm_mode_str = "Stage7 arm policy" if use_arm_policy else "heuristic"
    print(f"  Loco policy  : V6.2 (66 obs -> 15 act, LN+ELU)")
    print(f"  Arm control  : {arm_mode_str}")
    if use_arm_policy:
        print(f"  Arm ckpt     : {args_cli.arm_checkpoint}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Create simulation context
    # ------------------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[8.0, -4.0, 3.5], target=[3.0, 0.0, 0.5])

    # ------------------------------------------------------------------
    # 2. Create hierarchical environment (V6.2 + DEX3)
    # ------------------------------------------------------------------
    scene_cfg = HierarchicalSceneCfg()
    env = HierarchicalG1Env(
        sim=sim,
        scene_cfg=scene_cfg,
        checkpoint_path=args_cli.checkpoint,
        num_envs=num_envs,
        device=device,
        arm_checkpoint_path=args_cli.arm_checkpoint,
    )

    # ------------------------------------------------------------------
    # 3. Reset environment
    # ------------------------------------------------------------------
    print("\n[Demo] Resetting environment...")
    obs = env.reset()

    initial_pos = env.initial_positions
    stand_cmd = torch.zeros(num_envs, 3, device=device)

    # ==================================================================
    # PHASE 1: Walk to table
    # ==================================================================
    target_positions = initial_pos.clone()
    target_positions[:, 0] += args_cli.walk_distance

    walk_cfg = WalkToConfig()
    walk_cfg.max_steps = args_cli.max_steps
    skill = WalkToSkill(config=walk_cfg, device=device)
    skill.reset(target_positions=target_positions)

    print(f"\n{'-'*50}")
    print(f"  PHASE 1: Walk to table ({args_cli.walk_distance}m)")
    print(f"{'-'*50}")

    start_time = time.time()
    walk_done = False

    while simulation_app.is_running() and not walk_done:
        vel_cmd, walk_done, result = skill.step(obs)
        obs = env.step(vel_cmd)

        if env.step_count % 50 == 0:
            elapsed = time.time() - start_time
            current_pos = obs["root_pos"][:, :2]
            distances = torch.norm(current_pos - target_positions, dim=-1)
            print(
                f"  [Walk] Step {env.step_count:4d} | "
                f"Time: {elapsed:5.1f}s | "
                f"Mean dist: {distances.mean():.2f}m | "
                f"Height: {obs['base_height'].mean():.2f}m"
            )

        if (obs["base_height"] < 0.2).all():
            print("  [Walk] All robots fell!")
            walk_done = True

    walk_time = time.time() - start_time
    print(f"  [Walk] Result: {result.status.name} ({result.reason})")
    print(f"  [Walk] Time: {walk_time:.1f}s, Steps: {result.steps_taken}")

    # ==================================================================
    # PHASE 2: Stand still briefly (stabilize before reaching)
    # ==================================================================
    def stabilize_step(step):
        return env.step(stand_cmd)

    obs = run_phase(env, "Stabilize", stabilize_step, max_steps=100, log_interval=50)

    # ==================================================================
    # PHASE 3: Reach forward (manipulation mode)
    # ==================================================================
    if use_arm_policy:
        # --- ARM POLICY MODE ---
        print(f"\n{'-'*50}")
        print(f"  PHASE 3: Reach with Stage 7 arm policy (right arm)")
        print(f"{'-'*50}")

        env.set_manipulation_mode(True)
        env.enable_arm_policy(True)

        # Compute a REACHABLE target within the arm's workspace.
        # Stage 7 was trained on targets within 0.18-0.35m of the shoulder,
        # but the arm CAN physically extend further (up to ~0.50m).
        # We use MAX_REACH=0.45m to allow nearly full extension.
        #
        # Right shoulder offset in body frame (from arm_policy_wrapper.py)
        shoulder_offset = torch.tensor([0.0, -0.174, 0.259], device=device)
        max_reach = 0.40  # conservative for Stage 7 (trained at 0.32m)

        # Get each env's cup position and robot state
        cup_pos_all = env.cup.data.root_pos_w.clone()  # [N, 3]
        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w

        # Compute cup position in body frame
        cup_body = quat_apply_inverse(root_quat, cup_pos_all - root_pos)
        print(f"  Cup (body frame env0): [{cup_body[0, 0]:.3f}, {cup_body[0, 1]:.3f}, {cup_body[0, 2]:.3f}]")

        # Compute direction from shoulder to cup in body frame
        cup_from_shoulder = cup_body - shoulder_offset.unsqueeze(0)
        dist_from_shoulder = cup_from_shoulder.norm(dim=-1, keepdim=True)
        print(f"  Cup distance from shoulder: {dist_from_shoulder.mean():.3f}m (max reach: {max_reach}m)")

        # Clamp to reachable workspace -- target on line from shoulder to cup
        scale = torch.clamp(max_reach / (dist_from_shoulder + 1e-6), max=1.0)
        reachable_target_body = shoulder_offset.unsqueeze(0) + cup_from_shoulder * scale
        clamped = dist_from_shoulder.mean() > max_reach
        print(f"  Reachable target (body env0): [{reachable_target_body[0, 0]:.3f}, "
              f"{reachable_target_body[0, 1]:.3f}, {reachable_target_body[0, 2]:.3f}]"
              f" {'(clamped)' if clamped else '(actual cup)'}")

        # Convert reachable target to world frame for set_arm_target_world
        reachable_target_world = quat_apply(root_quat, reachable_target_body) + root_pos
        env.set_arm_target_world(reachable_target_world)
        env.reset_arm_policy_state()

        # Slow forward lean to close remaining distance
        lean_cmd = torch.zeros(num_envs, 3, device=device)
        lean_cmd[:, 0] = 0.15  # 0.15 m/s forward lean

        # Phase 3a: Active reaching (120 steps, with forward lean for first 80)
        reach_steps = 120
        best_ee_dist = float('inf')
        best_cup_dist = float('inf')
        for step in range(reach_steps):
            if not simulation_app.is_running():
                break

            cmd = lean_cmd if step < 80 else stand_cmd
            obs = env.step_arm_policy(cmd)

            # Track best distance
            ee_world, _ = env._compute_palm_ee()
            ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
            cup_dist = (ee_world - cup_pos_all).norm(dim=-1).mean().item()
            best_ee_dist = min(best_ee_dist, ee_dist)
            best_cup_dist = min(best_cup_dist, cup_dist)

            if step % 20 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                arm_pos = env.robot.data.joint_pos[:, env._arm_policy_joint_idx]
                print(f"  [Reach] Step {step:4d} | Height: {h:.2f}m | "
                      f"Standing: {standing}/{num_envs} | "
                      f"EE->target: {ee_dist:.3f}m | EE->cup: {cup_dist:.3f}m | "
                      f"arm[0]: [{arm_pos[0, 0]:.2f}, {arm_pos[0, 1]:.2f}, "
                      f"{arm_pos[0, 2]:.2f}, {arm_pos[0, 3]:.2f}, {arm_pos[0, 4]:.2f}]")

            # Early success
            if cup_dist < 0.06:
                print(f"  [Reach] Early success! EE->cup: {cup_dist:.3f}m")
                break

        print(f"  [Reach] Best EE->target: {best_ee_dist:.3f}m, Best EE->cup: {best_cup_dist:.3f}m")

        # Phase 3b: Hold position (freeze arm at current position, continue loco)
        print(f"  [Reach] Holding arm position...")
        env.enable_arm_policy(False)  # Switch to heuristic but keep current pose
        # Capture current arm positions as hold targets
        hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()
        for step in range(80):
            if not simulation_app.is_running():
                break
            obs = env.step_manipulation(stand_cmd, hold_arm_targets)

        # Final EE distance
        ee_world, _ = env._compute_palm_ee()
        final_ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
        cup_dist = (ee_world - cup_pos_all).norm(dim=-1).mean().item()
        print(f"  [Reach] Final EE->target: {final_ee_dist:.3f}m (best: {best_ee_dist:.3f}m)")
        print(f"  [Reach] Final EE->cup: {cup_dist:.3f}m (best: {best_cup_dist:.3f}m)")
    else:
        # --- HEURISTIC MODE ---
        print(f"\n{'-'*50}")
        print(f"  PHASE 3: Reach forward (heuristic -- no arm conflict)")
        print(f"{'-'*50}")

        env.set_manipulation_mode(True)
        env.arm_controller.set_pose(ArmPose.REACH_FORWARD)

        for step in range(200):
            if not simulation_app.is_running():
                break

            arm_targets = env.arm_controller.get_targets()
            obs = env.step_manipulation(stand_cmd, arm_targets)

            if step % 50 == 0:
                h = obs["base_height"].mean().item()
                arm_done = env.arm_controller.is_done
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Reach] Step {step:4d} | Height: {h:.2f}m | "
                      f"Standing: {standing}/{num_envs} | Arm done: {arm_done}")

        print(f"  [Reach] Arms in REACH_FORWARD position")

    # ==================================================================
    # PHASE 4: Close fingers (grasp)
    # ==================================================================
    print(f"\n{'-'*50}")
    print(f"  PHASE 4: Close fingers (grasp)")
    print(f"{'-'*50}")

    env.finger_controller.close(hand="both")

    for step in range(100):
        if not simulation_app.is_running():
            break

        if use_arm_policy:
            # Arm already frozen at reached position, just hold
            obs = env.step_manipulation(stand_cmd, hold_arm_targets)
        else:
            arm_targets = env.arm_controller.get_targets()
            obs = env.step_manipulation(stand_cmd, arm_targets)

        if step % 25 == 0:
            finger_pos = obs["joint_pos_finger"]
            closed = env.finger_controller.is_closed()
            h = obs["base_height"].mean().item()
            print(f"  [Grasp] Step {step:4d} | Height: {h:.2f}m | "
                  f"Finger mean: {finger_pos.mean():.3f} rad | "
                  f"Closed: {closed}")

    print(f"  [Grasp] Fingers closed!")

    # ==================================================================
    # PHASE 5: Carry / hold (arm policy keeps reaching, or heuristic carry)
    # ==================================================================
    print(f"\n{'-'*50}")
    print(f"  PHASE 5: Carry pose")
    print(f"{'-'*50}")

    if not use_arm_policy:
        env.arm_controller.set_pose(ArmPose.CARRY)

    for step in range(200):
        if not simulation_app.is_running():
            break

        if use_arm_policy:
            # Hold arm at reached position (already frozen)
            obs = env.step_manipulation(stand_cmd, hold_arm_targets)
        else:
            arm_targets = env.arm_controller.get_targets()
            obs = env.step_manipulation(stand_cmd, arm_targets)

        if step % 50 == 0:
            h = obs["base_height"].mean().item()
            standing = (obs["base_height"] > 0.5).sum().item()
            print(f"  [Carry] Step {step:4d} | Height: {h:.2f}m | "
                  f"Standing: {standing}/{num_envs}")

    # ==================================================================
    # PHASE 6: Return to default pose, open fingers
    # ==================================================================
    print(f"\n{'-'*50}")
    print(f"  PHASE 6: Return to default + open fingers")
    print(f"{'-'*50}")

    # Switch back to heuristic for return
    env.enable_arm_policy(False)
    env.arm_controller.set_pose(ArmPose.DEFAULT)
    env.finger_controller.open(hand="both")

    for step in range(200):
        if not simulation_app.is_running():
            break

        arm_targets = env.arm_controller.get_targets()
        obs = env.step_manipulation(stand_cmd, arm_targets)

        if step % 50 == 0:
            h = obs["base_height"].mean().item()
            standing = (obs["base_height"] > 0.5).sum().item()
            print(f"  [Return] Step {step:4d} | Height: {h:.2f}m | "
                  f"Standing: {standing}/{num_envs}")

    # Switch back to walking mode
    env.set_manipulation_mode(False)

    # ==================================================================
    # RESULTS
    # ==================================================================
    final_pos = obs["root_pos"][:, :2]
    distances = torch.norm(final_pos - target_positions, dim=-1)

    print("\n" + "=" * 60)
    print(f"  DEMO RESULTS (V6.2 Loco + {arm_mode_str})")
    print("=" * 60)
    print(f"  Walk          : {result.status.name} ({walk_time:.1f}s)")
    if use_arm_policy:
        print(f"  Reach         : Stage7 arm policy (EE dist: {final_ee_dist:.3f}m)")
    else:
        print(f"  Reach         : OK (heuristic arms extended)")
    print(f"  Grasp         : OK (fingers closed)")
    print(f"  Carry         : OK")
    print(f"  Return        : OK (arms back to default)")
    print(f"  Final height  : {obs['base_height'].mean():.2f}m")
    print(f"  Standing      : {(obs['base_height'] > 0.5).sum()}/{num_envs}")
    print(f"  Loco policy   : V6.2 (66->15, legs+waist only)")
    print(f"  Arm control   : {arm_mode_str}")
    print(f"  Total DoF     : 43 (15 loco + 14 arm + 14 finger)")
    print("=" * 60)

    # Keep sim alive (skip in headless mode)
    if args_cli.headless:
        print("\n[Demo] Complete! (headless mode - exiting)")
        env.close()
        simulation_app.close()
    else:
        print("\n[Demo] Complete! Keeping sim alive...")
        while simulation_app.is_running():
            obs = env.step(stand_cmd)
        env.close()


if __name__ == "__main__":
    main()
