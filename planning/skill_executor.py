"""
Skill Executor
================
Executes a plan (list of skill steps) sequentially on the HierarchicalG1Env.

Skills:
    - walk_to: WalkToSkill with stop_distance, then 50-step stabilize
    - reach: manipulation mode -> Stage 7 arm policy -> magnetic attach
    - grasp: finger_controller.close("both"), hold
    - lift: raise arm above basket height (modify shoulder_pitch)
    - lateral_walk: sidestep while holding arm position
    - place: detach object, open fingers, arm back to default
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
    MAX_REACH = 0.42  # Extended for table-top reach (Stage 7 trained at 0.32m)

    # Arm joint indices within the 14-joint arm group
    # ARM_JOINT_NAMES order: L_sh_pitch(0), L_sh_roll(1), L_sh_yaw(2), L_elbow(3),
    #   L_wr_roll(4), L_wr_pitch(5), L_wr_yaw(6),
    #   R_sh_pitch(7), R_sh_roll(8), R_sh_yaw(9), R_elbow(10),
    #   R_wr_roll(11), R_wr_pitch(12), R_wr_yaw(13)
    R_SHOULDER_PITCH = 7
    R_SHOULDER_ROLL = 8
    R_SHOULDER_YAW = 9
    R_ELBOW = 10

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
            "lift": self._execute_lift,
            "lateral_walk": self._execute_lateral_walk,
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
            # Keep manipulation mode -- arm stays at current (grasp) position
            print("  [WalkTo] Holding arm position during walk")
            env.set_manipulation_mode(True)
            env.enable_arm_policy(False)
            arm_targets = self._hold_arm_targets
        else:
            # Normal walking mode -- arm returns to default
            env.set_manipulation_mode(False)
            arm_targets = None

        # Configure WalkTo skill with stop_distance
        walk_cfg = WalkToConfig()
        walk_cfg.stop_distance = stop_distance
        walk_cfg.max_steps = 4000  # 80s at 50Hz
        if hold_arm:
            walk_cfg.max_forward_vel = 0.5
            walk_cfg.max_yaw_rate = 0.8
            walk_cfg.max_lateral_vel = 0.2
        else:
            walk_cfg.max_yaw_rate = 0.8
            walk_cfg.max_lateral_vel = 0.4

        skill = WalkToSkill(config=walk_cfg, device=str(self.device))
        skill.reset(target_positions=target_xy)

        # Execute walk loop
        obs = env.get_obs()
        walk_done = False
        start_time = time.time()

        while self._is_running() and not walk_done:
            vel_cmd, walk_done, result = skill.step(obs)

            if arm_targets is not None:
                obs = env.step_manipulation(vel_cmd, arm_targets)
            else:
                obs = env.step(vel_cmd)

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
    # reach: Extend arm to target using Stage 7 arm policy + magnetic attach
    # ------------------------------------------------------------------
    def _execute_reach(self, target: str) -> dict:
        """Reach toward a target with the Stage 7 arm policy.

        Strategy:
        1. Compute reachable target clamped to MAX_REACH from shoulder
        2. Apply Z correction to compensate for arm policy Z overshoot
        3. Set arm target and reach (no forward lean)
        4. Magnetic attach when EE within 0.30m of object
        5. Hold phase: freeze arm, continue loco for stability
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
            obj_pos_all = per_env_pos  # [num_envs, 3]
        else:
            target_pos = self.semantic_map.get_object_position(target)
            if target_pos is None:
                return {"status": "failed", "reason": f"Target '{target}' not found"}
            obj_pos_all = torch.tensor(
                [target_pos], dtype=torch.float32, device=self.device,
            ).expand(env.num_envs, -1)

        # Compute reachable target within arm workspace
        shoulder_offset = torch.tensor(self.SHOULDER_OFFSET, device=self.device)

        root_pos = env.robot.data.root_pos_w
        root_quat = env.robot.data.root_quat_w

        # Object in body frame
        obj_body = quat_apply_inverse(root_quat, obj_pos_all - root_pos)

        # Direction from shoulder to object
        obj_from_shoulder = obj_body - shoulder_offset.unsqueeze(0)
        dist_from_shoulder = obj_from_shoulder.norm(dim=-1, keepdim=True)

        # Debug: print world coordinates
        print(f"  [Reach] === DEBUG COORDINATES (env 0) ===")
        print(f"  [Reach]   Robot pos:    [{root_pos[0,0]:.3f}, {root_pos[0,1]:.3f}, {root_pos[0,2]:.3f}]")
        print(f"  [Reach]   Obj world:    [{obj_pos_all[0,0]:.3f}, {obj_pos_all[0,1]:.3f}, {obj_pos_all[0,2]:.3f}]")
        print(f"  [Reach]   Obj body:     [{obj_body[0,0]:.3f}, {obj_body[0,1]:.3f}, {obj_body[0,2]:.3f}]")
        print(f"  [Reach]   Shoulder:     [{shoulder_offset[0]:.3f}, {shoulder_offset[1]:.3f}, {shoulder_offset[2]:.3f}]")
        print(f"  [Reach]   Dist from shoulder: {dist_from_shoulder.mean():.3f}m (max: {self.MAX_REACH}m)")

        # Clamp XY distance from shoulder, preserve Z
        obj_from_shoulder_xy = obj_from_shoulder[:, :2]
        dist_xy = obj_from_shoulder_xy.norm(dim=-1, keepdim=True)
        scale_xy = torch.clamp(self.MAX_REACH / (dist_xy + 1e-6), max=1.0)

        reachable_target_body = torch.zeros_like(obj_body)
        reachable_target_body[:, :2] = shoulder_offset[:2].unsqueeze(0) + obj_from_shoulder_xy * scale_xy
        # Z: lower by 0.10m to compensate for arm policy Z overshoot
        reachable_target_body[:, 2] = obj_body[:, 2] - 0.10

        clamped = (dist_xy.mean() > self.MAX_REACH)
        if clamped:
            print(f"  [Reach] Target XY CLAMPED: {dist_xy.mean():.3f}m -> {self.MAX_REACH}m")
        else:
            print(f"  [Reach] Target within reach, using actual object position")

        print(f"  [Reach]   Reachable body: [{reachable_target_body[0,0]:.3f}, {reachable_target_body[0,1]:.3f}, {reachable_target_body[0,2]:.3f}]")
        print(f"  [Reach]   (Z lowered by 0.10m to compensate for overshoot)")

        # Convert to world frame and set as arm target
        reachable_target_world = quat_apply(root_quat, reachable_target_body) + root_pos
        print(f"  [Reach]   Reachable world: [{reachable_target_world[0,0]:.3f}, {reachable_target_world[0,1]:.3f}, {reachable_target_world[0,2]:.3f}]")

        env.set_arm_target_world(reachable_target_world)
        env.reset_arm_policy_state()

        # Stand still during reach (no forward lean -- object on table)
        reach_steps = 160
        best_obj_dist = float('inf')
        best_ee_dist = float('inf')
        attached_during_reach = False

        print(f"  [Reach] Starting active reach ({reach_steps} steps)...")
        for step in range(reach_steps):
            if not self._is_running():
                break

            obs = env.step_arm_policy(self._stand_cmd)

            # Track distances using LIVE positions
            ee_world, _ = env._compute_palm_ee()
            live_obj_pos = env.pickup_obj.data.root_pos_w
            ee_dist = (ee_world - env._arm_target_world).norm(dim=-1).mean().item()
            obj_dist = (ee_world - live_obj_pos).norm(dim=-1).mean().item()
            best_ee_dist = min(best_ee_dist, ee_dist)
            best_obj_dist = min(best_obj_dist, obj_dist)

            if step % 10 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                print(f"  [Reach] Step {step:3d} | h={h:.2f} | "
                      f"stand={standing}/{env.num_envs} | "
                      f"EE->tgt={ee_dist:.3f} | EE->obj={obj_dist:.3f} | "
                      f"EE=[{ee_world[0,0]:.2f},{ee_world[0,1]:.2f},{ee_world[0,2]:.2f}] "
                      f"Obj=[{live_obj_pos[0,0]:.2f},{live_obj_pos[0,1]:.2f},{live_obj_pos[0,2]:.2f}]")

            # Magnetic attach: 0.30m trigger, 0.40m max_dist
            if not attached_during_reach and obj_dist < 0.30:
                attached_during_reach = env.attach_object_to_hand(max_dist=0.40)
                if attached_during_reach:
                    print(f"  [Reach] ** Magnetic attach at step {step}! **")
                    break

            if obj_dist < 0.06:
                print(f"  [Reach] ** Early success! EE->obj: {obj_dist:.3f}m **")
                break

        print(f"  [Reach] Best EE->target: {best_ee_dist:.3f}m, Best EE->obj: {best_obj_dist:.3f}m")

        # Hold phase: freeze arm, continue loco for stability
        print("  [Reach] Holding arm position (50 steps)...")
        env.enable_arm_policy(False)
        self._hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()

        for step in range(50):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, self._hold_arm_targets)

        # Final distance check
        ee_world, _ = env._compute_palm_ee()
        live_obj_pos = env.pickup_obj.data.root_pos_w
        final_obj_dist = (ee_world - live_obj_pos).norm(dim=-1).mean().item()
        print(f"  [Reach] Final EE->obj: {final_obj_dist:.3f}m, attached={attached_during_reach}")

        return {
            "status": "success",
            "reason": f"Reached (best obj dist: {best_obj_dist:.3f}m, attached={attached_during_reach})",
            "attached": attached_during_reach,
        }

    # ------------------------------------------------------------------
    # grasp: Close fingers + magnetic attach
    # ------------------------------------------------------------------
    def _execute_grasp(self) -> dict:
        """Close fingers and magnetically attach object to palm.

        1. Close fingers (visual)
        2. Try magnetic attach (snap object to palm if close enough)
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
            print("  [Grasp] Object already attached from reach phase")

        # Close fingers for 30 steps
        for step in range(30):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

        # Magnetic attach (skip if already attached)
        if not already_attached:
            attached = env.attach_object_to_hand(max_dist=0.40)
        else:
            attached = True

        # Hold for 50 more steps
        for step in range(50):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, arm_targets)

            if step % 25 == 0:
                h = obs["base_height"].mean().item()
                print(f"  [Grasp] Step {step:4d} | Height: {h:.2f}m | Attached: {attached}")

        if attached:
            return {"status": "success", "reason": "Object attached to hand"}
        else:
            return {"status": "failed", "reason": "Could not attach object (too far)"}

    # ------------------------------------------------------------------
    # lift: Raise arm above basket height
    # ------------------------------------------------------------------
    def _execute_lift(self) -> dict:
        """Lift the held object above basket height by adjusting arm joints.

        Modifies right shoulder pitch and elbow to raise the hand.
        Interpolates smoothly over 80 steps to avoid destabilizing the robot.
        """
        env = self.env

        if self._hold_arm_targets is None:
            return {"status": "failed", "reason": "No arm targets to lift from"}

        start_targets = self._hold_arm_targets.clone()
        lift_targets = self._hold_arm_targets.clone()

        # Set lifted position for right arm:
        # shoulder_pitch ~0.1 (raised above horizontal)
        # shoulder_roll stays (keep arm to the side)
        # elbow ~0.5 (slightly bent to keep object forward)
        lift_targets[:, self.R_SHOULDER_PITCH] = 0.10
        lift_targets[:, self.R_ELBOW] = 0.50

        print(f"  [Lift] Raising arm: shoulder_pitch -> 0.10, elbow -> 0.50")

        # Smoothly interpolate over 80 steps
        for step in range(80):
            if not self._is_running():
                break
            alpha = min(1.0, step / 50.0)  # Ramp over 50 steps
            interp = start_targets * (1 - alpha) + lift_targets * alpha
            obs = env.step_manipulation(self._stand_cmd, interp)

            if step % 20 == 0:
                h = obs["base_height"].mean().item()
                ee_world, _ = env._compute_palm_ee()
                print(f"  [Lift] Step {step:3d} | h={h:.2f} | "
                      f"EE z={ee_world[0,2]:.3f} | alpha={alpha:.2f}")

        self._hold_arm_targets = lift_targets.clone()

        # Final EE height
        ee_world, _ = env._compute_palm_ee()
        print(f"  [Lift] Final EE: [{ee_world[0,0]:.3f}, {ee_world[0,1]:.3f}, {ee_world[0,2]:.3f}]")

        return {"status": "success", "reason": f"Lifted to EE z={ee_world[0,2].item():.3f}m"}

    # ------------------------------------------------------------------
    # lateral_walk: Sidestep while holding arm position
    # ------------------------------------------------------------------
    def _execute_lateral_walk(self, direction: str = "right", distance: float = 0.4, speed: float = 0.25) -> dict:
        """Walk laterally while holding the arm in position.

        Args:
            direction: "right" or "left" (robot's perspective)
            distance: meters to walk sideways
            speed: lateral velocity (m/s)
        """
        env = self.env

        if self._hold_arm_targets is None:
            return {"status": "failed", "reason": "No arm targets held"}

        # Lateral velocity command (negative Y = right in body frame)
        vy = -speed if direction == "right" else speed
        lateral_cmd = torch.zeros(env.num_envs, 3, device=self.device)
        lateral_cmd[:, 1] = vy

        # Steps: distance / speed / control_dt
        steps = int(distance / speed / 0.02)  # 0.02s per step at 50Hz

        print(f"  [Lateral] Walking {direction} {distance}m at {speed}m/s ({steps} steps)")

        for step in range(steps):
            if not self._is_running():
                break
            obs = env.step_manipulation(lateral_cmd, self._hold_arm_targets)

            if step % 50 == 0:
                h = obs["base_height"].mean().item()
                standing = (obs["base_height"] > 0.5).sum().item()
                ee_world, _ = env._compute_palm_ee()
                print(f"  [Lateral] Step {step}/{steps} | h={h:.2f} | "
                      f"stand={standing}/{env.num_envs} | "
                      f"EE=[{ee_world[0,0]:.2f},{ee_world[0,1]:.2f},{ee_world[0,2]:.2f}]")

        # Brief stabilize
        for _ in range(30):
            if not self._is_running():
                break
            obs = env.step_manipulation(self._stand_cmd, self._hold_arm_targets)

        return {"status": "success", "reason": f"Walked {direction} ~{distance}m"}

    # ------------------------------------------------------------------
    # place: Release object, open fingers, return arm to default
    # ------------------------------------------------------------------
    def _execute_place(self) -> dict:
        """Detach object, open fingers, return arm to default.

        1. Detach object (drops under gravity)
        2. Open fingers
        3. Return arm to default pose (heuristic)
        4. 200-step transition
        5. Switch back to walking mode
        """
        from ..low_level.arm_controller import ArmPose

        env = self.env

        # Detach object (magnetic grasp release)
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
