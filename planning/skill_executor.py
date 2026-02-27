"""
Skill Executor
================
Executes a plan (list of skill steps) sequentially on the HierarchicalG1Env.

Reuses the exact control patterns from test_hierarchical.py:
    - walk_to: WalkToSkill with stop_distance, then 50-step stabilize
    - reach: manipulation mode → Stage 7 arm policy → 80-step active reach + hold
    - grasp: finger_controller.close("both"), 100-step hold
    - place: finger_controller.open("both"), arm back to default, 200 steps
"""

from __future__ import annotations

import math
import time
import torch
from typing import Any, Optional

from .semantic_map import SemanticMap


class SkillExecutor:
    """Execute a skill plan on the HierarchicalG1Env.

    Args:
        env: HierarchicalG1Env instance
        semantic_map: SemanticMap instance for real-time position queries
        simulation_app: IsaacSim app handle (for is_running() check)
    """

    # Right shoulder offset in body frame (from arm_policy_wrapper.py)
    SHOULDER_OFFSET = [0.0, -0.174, 0.259]
    MAX_REACH = 0.32  # Stage 7 training workspace radius

    def __init__(
        self,
        env: Any,
        semantic_map: SemanticMap,
        simulation_app: Any = None,
    ):
        self.env = env
        self.semantic_map = semantic_map
        self.sim_app = simulation_app
        self.device = env.device

        self._stand_cmd = torch.zeros(env.num_envs, 3, device=self.device)
        self._hold_arm_targets: Optional[torch.Tensor] = None

        # Skill dispatch table
        self._skills = {
            "walk_to": self._execute_walk_to,
            "reach": self._execute_reach,
            "grasp": self._execute_grasp,
            "place": self._execute_place,
            "walk_to_position": self._execute_walk_to_position,
        }

    def _is_running(self) -> bool:
        """Check if simulation is still running."""
        if self.sim_app is not None:
            return self.sim_app.is_running()
        return True

    def execute_plan(self, plan: list) -> dict:
        """Execute a sequence of skill steps.

        Args:
            plan: List of {"skill": str, "params": dict}

        Returns:
            {"plan_results": [...], "completed": bool}
        """
        results = []
        print(f"\n{'='*60}")
        print(f"  EXECUTING PLAN ({len(plan)} steps)")
        print(f"{'='*60}")

        for i, step in enumerate(plan):
            if not self._is_running():
                break

            skill_name = step["skill"]
            params = step.get("params", {})

            print(f"\n{'-'*50}")
            print(f"  Step {i+1}/{len(plan)}: {skill_name}({params})")
            print(f"{'-'*50}")

            # Update semantic map for latest positions
            self.semantic_map.update()

            # Dispatch skill
            handler = self._skills.get(skill_name)
            if handler is None:
                result = {"status": "failed", "reason": f"Unknown skill: {skill_name}"}
            else:
                result = handler(**params)

            results.append({"skill": skill_name, "params": params, "result": result})
            print(f"  → {skill_name}: {result['status']} ({result.get('reason', '')})")

            if result["status"] == "failed":
                print(f"\n  [Executor] PLAN FAILED at step {i+1}")
                break

        completed = all(r["result"]["status"] == "success" for r in results)
        print(f"\n{'='*60}")
        print(f"  PLAN {'COMPLETED' if completed else 'INCOMPLETE'}")
        print(f"  Results: {sum(1 for r in results if r['result']['status'] == 'success')}/{len(results)} succeeded")
        print(f"{'='*60}")

        return {"plan_results": results, "completed": completed}

    # ------------------------------------------------------------------
    # walk_to: Navigate to object/surface using WalkToSkill
    # ------------------------------------------------------------------
    def _execute_walk_to(self, target: str, stop_distance: float = 0.25) -> dict:
        """Walk to an object or surface, stopping at stop_distance.

        Uses WalkToSkill with the semantic map position.
        Stabilizes for 50 steps after arrival.
        """
        from ..skills.walk_to import WalkToSkill
        from ..config.skill_config import WalkToConfig

        # Get target position from semantic map
        target_pos = self.semantic_map.get_position(target)
        if target_pos is None:
            return {"status": "failed", "reason": f"Target '{target}' not found in semantic map"}

        target_xy = torch.tensor(
            [[target_pos[0], target_pos[1]]],
            dtype=torch.float32,
            device=self.device,
        ).expand(self.env.num_envs, -1)

        # Ensure walking mode
        self.env.set_manipulation_mode(False)

        # Configure WalkTo skill with stop_distance
        walk_cfg = WalkToConfig()
        walk_cfg.stop_distance = stop_distance
        walk_cfg.max_steps = 2000

        skill = WalkToSkill(config=walk_cfg, device=str(self.device))
        skill.reset(target_positions=target_xy)

        # Execute walk loop
        obs = self.env.get_obs()
        walk_done = False
        start_time = time.time()

        while self._is_running() and not walk_done:
            vel_cmd, walk_done, result = skill.step(obs)
            obs = self.env.step(vel_cmd)

            # Safety: all robots fell
            if (obs["base_height"] < 0.2).all():
                return {"status": "failed", "reason": "All robots fell during walk"}

        walk_time = time.time() - start_time
        print(f"  [WalkTo] {result.status.name} in {walk_time:.1f}s, {result.steps_taken} steps")

        # Stabilize after walk
        print("  [WalkTo] Stabilizing...")
        for _ in range(50):
            if not self._is_running():
                break
            obs = self.env.step(self._stand_cmd)

        if result.succeeded:
            return {"status": "success", "reason": f"Reached within {stop_distance}m of {target}"}
        else:
            return {"status": "failed", "reason": f"Walk failed: {result.reason}"}

    # ------------------------------------------------------------------
    # walk_to_position: Navigate to specific world coordinates
    # ------------------------------------------------------------------
    def _execute_walk_to_position(self, x: float, y: float, stop_distance: float = 0.3) -> dict:
        """Walk to specific XY world coordinates."""
        from ..skills.walk_to import WalkToSkill
        from ..config.skill_config import WalkToConfig

        target_xy = torch.tensor(
            [[x, y]], dtype=torch.float32, device=self.device,
        ).expand(self.env.num_envs, -1)

        self.env.set_manipulation_mode(False)

        walk_cfg = WalkToConfig()
        walk_cfg.stop_distance = stop_distance
        walk_cfg.max_steps = 2000

        skill = WalkToSkill(config=walk_cfg, device=str(self.device))
        skill.reset(target_positions=target_xy)

        obs = self.env.get_obs()
        walk_done = False

        while self._is_running() and not walk_done:
            vel_cmd, walk_done, result = skill.step(obs)
            obs = self.env.step(vel_cmd)
            if (obs["base_height"] < 0.2).all():
                return {"status": "failed", "reason": "All robots fell"}

        # Stabilize
        for _ in range(50):
            if not self._is_running():
                break
            obs = self.env.step(self._stand_cmd)

        if result.succeeded:
            return {"status": "success", "reason": f"Reached ({x:.1f}, {y:.1f})"}
        else:
            return {"status": "failed", "reason": f"Walk failed: {result.reason}"}

    # ------------------------------------------------------------------
    # reach: Extend arm to target using Stage 7 arm policy
    # ------------------------------------------------------------------
    def _execute_reach(self, target: str) -> dict:
        """Reach toward a target with the Stage 7 arm policy.

        Pattern from test_hierarchical.py Phase 3:
        1. Compute reachable target (clamped to 0.32m from shoulder)
        2. Active reaching: 80 steps with arm policy
        3. Hold phase: freeze arm, continue loco for stability
        """
        from isaaclab.utils.math import quat_apply_inverse, quat_apply

        env = self.env

        if env.arm_policy is None:
            return {"status": "failed", "reason": "No arm policy loaded"}

        # Switch to manipulation mode + arm policy
        env.set_manipulation_mode(True)
        env.enable_arm_policy(True)

        # Get target position (refresh from semantic map)
        self.semantic_map.update()
        target_pos = self.semantic_map.get_object_position(target)
        if target_pos is None:
            return {"status": "failed", "reason": f"Target '{target}' not found"}

        # Compute reachable target within arm workspace
        # (exact pattern from test_hierarchical.py lines 226-251)
        shoulder_offset = torch.tensor(self.SHOULDER_OFFSET, device=self.device)

        cup_pos_all = torch.tensor(
            [target_pos], dtype=torch.float32, device=self.device,
        ).expand(env.num_envs, -1)

        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w

        # Cup in body frame
        cup_body = quat_apply_inverse(root_quat, cup_pos_all - root_pos)

        # Direction from shoulder to cup, clamped to max_reach
        cup_from_shoulder = cup_body - shoulder_offset.unsqueeze(0)
        dist_from_shoulder = cup_from_shoulder.norm(dim=-1, keepdim=True)
        print(f"  [Reach] Target distance from shoulder: {dist_from_shoulder.mean():.3f}m (max: {self.MAX_REACH}m)")

        scale = torch.clamp(self.MAX_REACH / (dist_from_shoulder + 1e-6), max=1.0)
        reachable_target_body = shoulder_offset.unsqueeze(0) + cup_from_shoulder * scale

        # Convert to world frame
        reachable_target_world = quat_apply(root_quat, reachable_target_body) + root_pos
        env.set_arm_target_world(reachable_target_world)
        env.reset_arm_policy_state()

        # Phase A: Active reaching (80 steps)
        reach_steps = 80
        best_ee_dist = float('inf')

        for step in range(reach_steps):
            if not self._is_running():
                break
            obs = env.step_arm_policy(self._stand_cmd)

            ee_world, _ = env._compute_palm_ee()
            ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
            best_ee_dist = min(best_ee_dist, ee_dist)

            if step % 20 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Reach] Step {step:4d} | Height: {h:.2f}m | "
                      f"Standing: {standing}/{env.num_envs} | EE dist: {ee_dist:.3f}m")

        print(f"  [Reach] Best EE dist: {best_ee_dist:.3f}m")

        # Phase B: Hold position (freeze arm, continue loco)
        print("  [Reach] Holding arm position...")
        env.enable_arm_policy(False)
        self._hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        for step in range(100):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, self._hold_arm_targets)

        # Final distance
        ee_world, _ = env._compute_palm_ee()
        final_ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
        cup_dist = (ee_world - cup_pos_all).norm(dim=-1).mean().item()
        print(f"  [Reach] Final EE→target: {final_ee_dist:.3f}m, EE→cup: {cup_dist:.3f}m")

        return {
            "status": "success",
            "reason": f"Reached (best: {best_ee_dist:.3f}m, final: {final_ee_dist:.3f}m)",
            "best_ee_dist": best_ee_dist,
            "final_ee_dist": final_ee_dist,
            "cup_dist": cup_dist,
        }

    # ------------------------------------------------------------------
    # grasp: Close fingers
    # ------------------------------------------------------------------
    def _execute_grasp(self) -> dict:
        """Close fingers to grasp. 100-step hold for secure grip.

        Pattern from test_hierarchical.py Phase 4.
        """
        env = self.env
        env.finger_controller.close(hand="both")

        arm_targets = self._hold_arm_targets
        if arm_targets is None:
            arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        for step in range(100):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

            if step % 25 == 0:
                finger_pos = obs["joint_pos_finger"]
                closed = env.finger_controller.is_closed()
                h = obs["base_height"].mean().item()
                print(f"  [Grasp] Step {step:4d} | Height: {h:.2f}m | "
                      f"Finger mean: {finger_pos.mean():.3f} | Closed: {closed}")

        closed = env.finger_controller.is_closed()
        if closed:
            return {"status": "success", "reason": "Fingers closed"}
        else:
            return {"status": "success", "reason": "Grasp commanded (fingers closing)"}

    # ------------------------------------------------------------------
    # place: Open fingers and return arm to default
    # ------------------------------------------------------------------
    def _execute_place(self) -> dict:
        """Open fingers and return arm to default pose.

        Pattern from test_hierarchical.py Phase 6:
        1. Open fingers
        2. Return arm to default pose (heuristic)
        3. 200-step transition
        4. Switch back to walking mode
        """
        from ..low_level.arm_controller import ArmPose

        env = self.env

        # Switch to heuristic arm (default pose)
        env.enable_arm_policy(False)
        env.arm_controller.set_pose(ArmPose.DEFAULT)
        env.finger_controller.open(hand="both")

        for step in range(200):
            if not self._is_running():
                break
            arm_targets = env.arm_controller.get_targets()
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

            if step % 50 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Place] Step {step:4d} | Height: {h:.2f}m | "
                      f"Standing: {standing}/{env.num_envs}")

        # Switch back to walking mode
        env.set_manipulation_mode(False)
        self._hold_arm_targets = None

        return {"status": "success", "reason": "Object released, arm returned to default"}
