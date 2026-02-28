"""
Skill Executor
================
Executes a plan (list of skill steps) sequentially on the HierarchicalG1Env.

Reuses the exact control patterns from test_hierarchical.py:
    - walk_to: WalkToSkill with stop_distance, then 50-step stabilize
    - reach: manipulation mode -> Stage 7 arm policy -> 80-step active reach + hold
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
    MAX_REACH = 0.35  # Conservative for Stage 7 (trained at 0.32m), safe OOD extension

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
            print(f"  -> {skill_name}: {result['status']} ({result.get('reason', '')})")

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
    def _execute_walk_to(self, target: str, stop_distance: float = 0.25, hold_arm: bool = False) -> dict:
        """Walk to an object or surface, stopping at stop_distance.

        Uses WalkToSkill with the semantic map position.
        If hold_arm=True, keeps arm at current position (manipulation mode)
        instead of switching to walking mode (which resets arm to default).
        """
        from ..skills.walk_to import WalkToSkill
        from ..config.skill_config import WalkToConfig

        # Get per-env target positions (each env has objects at different world positions)
        per_env_pos = self.semantic_map.get_per_env_position(target)
        if per_env_pos is not None:
            target_xy = per_env_pos[:, :2]  # [num_envs, 2]
        else:
            # Fallback: single position expanded to all envs
            target_pos = self.semantic_map.get_position(target)
            if target_pos is None:
                return {"status": "failed", "reason": f"Target '{target}' not found in semantic map"}
            target_xy = torch.tensor(
                [[target_pos[0], target_pos[1]]],
                dtype=torch.float32,
                device=self.device,
            ).expand(self.env.num_envs, -1)

        env = self.env

        if hold_arm and self._hold_arm_targets is not None:
            # Keep manipulation mode — arm stays at current (grasp) position
            print("  [WalkTo] Holding arm position during walk")
            env.set_manipulation_mode(True)
            env.enable_arm_policy(False)
            arm_targets = self._hold_arm_targets
        else:
            # Normal walking mode — arm returns to default
            env.set_manipulation_mode(False)
            arm_targets = None

        # Configure WalkTo skill with stop_distance
        walk_cfg = WalkToConfig()
        walk_cfg.stop_distance = stop_distance
        walk_cfg.max_steps = 4000  # 80s at 50Hz — enough for 180-degree turns
        if hold_arm:
            # Slower, more stable walking when carrying object
            walk_cfg.max_forward_vel = 0.5
            walk_cfg.max_yaw_rate = 0.8  # Need fast yaw for 180-degree turns
            walk_cfg.max_lateral_vel = 0.2
        else:
            walk_cfg.max_yaw_rate = 0.8  # rad/s — faster turning for large direction changes
            walk_cfg.max_lateral_vel = 0.4  # m/s — faster lateral correction

        skill = WalkToSkill(config=walk_cfg, device=str(self.device))
        skill.reset(target_positions=target_xy)

        # Execute walk loop
        obs = env.get_obs()
        walk_done = False
        start_time = time.time()

        while self._is_running() and not walk_done:
            vel_cmd, walk_done, result = skill.step(obs)

            if arm_targets is not None:
                # Walk in manipulation mode, holding arm
                obs = env.step_manipulation(vel_cmd, arm_targets)
            else:
                # Normal walking mode
                obs = env.step(vel_cmd)

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
            if arm_targets is not None:
                obs = env.step_manipulation(self._stand_cmd, arm_targets)
            else:
                obs = env.step(self._stand_cmd)

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
        walk_cfg.max_steps = 4000
        walk_cfg.max_yaw_rate = 0.8
        walk_cfg.max_lateral_vel = 0.4

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

        Strategy (no lift phase - direct reach):
        1. Compute reachable target clamped to MAX_REACH from shoulder
        2. Set arm target and reach directly (no intermediate lift)
        3. Slow forward lean for first 80 steps to close last cm
        4. Track EE-to-cup distance, magnetic attach when close
        5. Hold phase: freeze arm, continue loco for stability

        Debug: prints world coordinates of EE, target, and cup each 10 steps.
        Draws colored markers if env.enable_debug_markers(True) was called.
        """
        from isaaclab.utils.math import quat_apply_inverse, quat_apply

        env = self.env

        if env.arm_policy is None:
            return {"status": "failed", "reason": "No arm policy loaded"}

        # Enable debug visualization markers
        env.enable_debug_markers(True)

        # Switch to manipulation mode + arm policy
        env.set_manipulation_mode(True)
        env.enable_arm_policy(True)

        # Get per-env target position (refresh from semantic map)
        self.semantic_map.update()
        per_env_pos = self.semantic_map.get_per_env_position(target)
        if per_env_pos is not None:
            cup_pos_all = per_env_pos  # [num_envs, 3]
        else:
            target_pos = self.semantic_map.get_object_position(target)
            if target_pos is None:
                return {"status": "failed", "reason": f"Target '{target}' not found"}
            cup_pos_all = torch.tensor(
                [target_pos], dtype=torch.float32, device=self.device,
            ).expand(env.num_envs, -1)

        # Compute reachable target within arm workspace
        shoulder_offset = torch.tensor(self.SHOULDER_OFFSET, device=self.device)

        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w

        # Cup in body frame
        cup_body = quat_apply_inverse(root_quat, cup_pos_all - root_pos)

        # Direction from shoulder to cup
        cup_from_shoulder = cup_body - shoulder_offset.unsqueeze(0)
        dist_from_shoulder = cup_from_shoulder.norm(dim=-1, keepdim=True)

        # Debug: print world coordinates
        print(f"  [Reach] === DEBUG COORDINATES (env 0) ===")
        print(f"  [Reach]   Robot pos:    [{root_pos[0,0]:.3f}, {root_pos[0,1]:.3f}, {root_pos[0,2]:.3f}]")
        print(f"  [Reach]   Cup world:    [{cup_pos_all[0,0]:.3f}, {cup_pos_all[0,1]:.3f}, {cup_pos_all[0,2]:.3f}]")
        print(f"  [Reach]   Cup body:     [{cup_body[0,0]:.3f}, {cup_body[0,1]:.3f}, {cup_body[0,2]:.3f}]")
        print(f"  [Reach]   Shoulder:     [{shoulder_offset[0]:.3f}, {shoulder_offset[1]:.3f}, {shoulder_offset[2]:.3f}]")
        print(f"  [Reach]   Dist from shoulder: {dist_from_shoulder.mean():.3f}m (max: {self.MAX_REACH}m)")

        # Clamp to MAX_REACH
        scale = torch.clamp(self.MAX_REACH / (dist_from_shoulder + 1e-6), max=1.0)
        reachable_target_body = shoulder_offset.unsqueeze(0) + cup_from_shoulder * scale

        # If cup is within MAX_REACH, target IS the cup
        clamped = (dist_from_shoulder.mean() > self.MAX_REACH)
        if clamped:
            print(f"  [Reach] Target CLAMPED: {dist_from_shoulder.mean():.3f}m -> {self.MAX_REACH}m")
        else:
            print(f"  [Reach] Target within reach, using actual cup position")

        print(f"  [Reach]   Reachable body: [{reachable_target_body[0,0]:.3f}, {reachable_target_body[0,1]:.3f}, {reachable_target_body[0,2]:.3f}]")

        # Convert to world frame and set as arm target
        reachable_target_world = quat_apply(root_quat, reachable_target_body) + root_pos
        print(f"  [Reach]   Reachable world: [{reachable_target_world[0,0]:.3f}, {reachable_target_world[0,1]:.3f}, {reachable_target_world[0,2]:.3f}]")

        env.set_arm_target_world(reachable_target_world)
        env.reset_arm_policy_state()

        # Slow forward lean velocity: 0.15 m/s helps close the last few cm
        lean_cmd = torch.zeros(env.num_envs, 3, device=self.device)
        lean_cmd[:, 0] = 0.15  # slow forward

        # Active reaching (120 steps: 80 with lean + 40 stabilize)
        reach_steps = 120
        best_cup_dist = float('inf')
        best_ee_dist = float('inf')
        attached_during_reach = False

        print(f"  [Reach] Starting active reach ({reach_steps} steps)...")
        for step in range(reach_steps):
            if not self._is_running():
                break

            # Use lean velocity for first 80 steps, then stand still to stabilize
            cmd = lean_cmd if step < 80 else self._stand_cmd
            obs = env.step_arm_policy(cmd)

            # Track distances using LIVE positions
            ee_world, _ = env._compute_palm_ee()
            live_cup_pos = env.cup.data.root_pos_w
            ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
            cup_dist = (ee_world - live_cup_pos).norm(dim=-1).mean().item()
            best_ee_dist = min(best_ee_dist, ee_dist)
            best_cup_dist = min(best_cup_dist, cup_dist)

            if step % 10 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                # Detailed debug with world coordinates
                print(f"  [Reach] Step {step:3d} | h={h:.2f} | "
                      f"stand={standing}/{env.num_envs} | "
                      f"EE->tgt={ee_dist:.3f} | EE->cup={cup_dist:.3f} | "
                      f"EE=[{ee_world[0,0]:.2f},{ee_world[0,1]:.2f},{ee_world[0,2]:.2f}] "
                      f"Cup=[{live_cup_pos[0,0]:.2f},{live_cup_pos[0,1]:.2f},{live_cup_pos[0,2]:.2f}]")

            # Try magnetic attach as soon as EE is close enough
            if not attached_during_reach and cup_dist < 0.15:
                attached_during_reach = env.attach_object_to_hand(max_dist=0.20)
                if attached_during_reach:
                    print(f"  [Reach] ** Magnetic attach at step {step}! **")
                    break

            # Early success: EE is very close to actual cup
            if cup_dist < 0.06:
                print(f"  [Reach] ** Early success! EE->cup: {cup_dist:.3f}m **")
                break

        print(f"  [Reach] Best EE->target: {best_ee_dist:.3f}m, Best EE->cup: {best_cup_dist:.3f}m")

        # Hold phase: freeze arm, continue loco for stability
        print("  [Reach] Holding arm position (80 steps)...")
        env.enable_arm_policy(False)
        self._hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        for step in range(80):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, self._hold_arm_targets)

        # Final distance check
        ee_world, _ = env._compute_palm_ee()
        live_cup_pos = env.cup.data.root_pos_w
        final_ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
        final_cup_dist = (ee_world - live_cup_pos).norm(dim=-1).mean().item()
        print(f"  [Reach] Final EE->target: {final_ee_dist:.3f}m, EE->cup(live): {final_cup_dist:.3f}m")
        print(f"  [Reach]   EE final:  [{ee_world[0,0]:.3f}, {ee_world[0,1]:.3f}, {ee_world[0,2]:.3f}]")
        print(f"  [Reach]   Cup final: [{live_cup_pos[0,0]:.3f}, {live_cup_pos[0,1]:.3f}, {live_cup_pos[0,2]:.3f}]")

        return {
            "status": "success",
            "reason": f"Reached (best cup dist: {best_cup_dist:.3f}m, final: {final_cup_dist:.3f}m)",
            "best_ee_dist": best_ee_dist,
            "best_cup_dist": best_cup_dist,
            "final_ee_dist": final_ee_dist,
            "cup_dist": final_cup_dist,
            "attached": attached_during_reach,
        }

    # ------------------------------------------------------------------
    # grasp: Close fingers
    # ------------------------------------------------------------------
    def _execute_grasp(self) -> dict:
        """Close fingers and magnetically attach cup to palm.

        1. Close fingers (visual)
        2. Try magnetic attach (snap cup to palm if close enough)
           - Skipped if already attached during reach phase
        3. Hold for 50 steps to stabilize
        """
        env = self.env
        env.finger_controller.close(hand="both")

        arm_targets = self._hold_arm_targets
        if arm_targets is None:
            arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        # Check if already attached during reach
        already_attached = getattr(env, '_object_attached', False)
        if already_attached:
            print("  [Grasp] Cup already attached from reach phase")

        # Close fingers for 30 steps
        for step in range(30):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

        # Magnetic attach: snap cup to palm (skip if already attached)
        if not already_attached:
            attached = env.attach_object_to_hand(max_dist=0.20)
        else:
            attached = True

        # Hold for 50 more steps
        for step in range(50):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

            if step % 25 == 0:
                finger_pos = obs["joint_pos_finger"]
                h = obs["base_height"].mean().item()
                print(f"  [Grasp] Step {step:4d} | Height: {h:.2f}m | "
                      f"Finger mean: {finger_pos.mean():.3f} | Attached: {attached}")

        if attached:
            return {"status": "success", "reason": "Cup attached to hand"}
        else:
            return {"status": "success", "reason": "Fingers closed (cup not attached)"}

    # ------------------------------------------------------------------
    # place: Open fingers and return arm to default
    # ------------------------------------------------------------------
    def _execute_place(self) -> dict:
        """Detach cup, open fingers, return arm to default.

        1. Detach cup (drops under gravity)
        2. Open fingers
        3. Return arm to default pose (heuristic)
        4. 200-step transition
        5. Switch back to walking mode
        """
        from ..low_level.arm_controller import ArmPose

        env = self.env

        # Detach cup (magnetic grasp release)
        env.detach_object()

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
