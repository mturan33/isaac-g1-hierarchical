"""
Stage 7 Arm Policy Wrapper
===========================
Loads the Stage 7 arm actor (right arm only, 55 obs -> 5 act)
and runs inference for the hierarchical control pipeline.

The arm actor was trained on the OLD 23-DoF G1 robot with these joint names:
  right_shoulder_pitch_joint, right_shoulder_roll_joint,
  right_shoulder_yaw_joint, right_elbow_pitch_joint, right_elbow_roll_joint

On the NEW 29-DoF G1+DEX3 robot, these map to:
  right_shoulder_pitch_joint  (same)
  right_shoulder_roll_joint   (same)
  right_shoulder_yaw_joint    (same)
  right_elbow_joint           (was: right_elbow_pitch_joint)
  right_wrist_roll_joint      (was: right_elbow_roll_joint)

The remaining wrist joints (wrist_pitch, wrist_yaw) are NOT controlled by
the policy and must be managed by the heuristic fallback.

Network: 55 -> [256, ELU, 256, ELU, 128, ELU] -> 5 (NO LayerNorm)
Action scale: 0.5  (target = default_arm + action * 0.5)
Default arm: ALL ZEROS (as trained on 23-DoF)

Checkpoint format (DualActorCritic):
  model_state_dict:
    arm_actor.net.0.weight  [256, 55]
    arm_actor.net.0.bias    [256]
    arm_actor.net.2.weight  [256, 256]
    arm_actor.net.2.bias    [256]
    arm_actor.net.4.weight  [128, 256]
    arm_actor.net.4.bias    [128]
    arm_actor.net.6.weight  [5, 128]
    arm_actor.net.6.bias    [5]
    arm_actor.log_std       [5]
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional


# Stage 7 arm obs/act dimensions
ARM_OBS_DIM = 55
ARM_ACT_DIM = 5

# Action scale (must match training)
ARM_ACTION_SCALE = 0.5

# Default arm pose — all zeros (as in 23-DoF training)
ARM_DEFAULT = [0.0, 0.0, 0.0, 0.0, 0.0]

# Palm forward offset for EE computation
PALM_FORWARD_OFFSET = 0.08

# Shoulder offset in body frame (right shoulder)
SHOULDER_OFFSET = [0.0, -0.174, 0.259]

# Stage 7 curriculum level 7 thresholds
OBS_POS_THRESHOLD = 0.04  # meters
MAX_REACH_STEPS = 150

# Joint limits from 23-DoF training robot (must clamp to prevent OOD)
# Order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
# Conservative limits (tighter than physical) to stay within training distribution
ARM_JOINT_LOWER = [-1.50, -1.50, -1.80, -0.20, -1.50]
ARM_JOINT_UPPER = [ 1.50,  1.00,  1.80,  2.50,  1.50]
HEIGHT_DEFAULT = 0.72     # Stage 7 training height command

# Joint name mapping: Stage 7 (23-DoF) -> 29-DoF
# Stage 7 arm joints -> 29-DoF equivalents
STAGE7_TO_29DOF = {
    "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint": "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}

# The 5 arm joints controlled by Stage 7 policy, in 29-DoF names
ARM_POLICY_JOINT_NAMES_29DOF = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",        # was: right_elbow_pitch_joint
    "right_wrist_roll_joint",   # was: right_elbow_roll_joint
]

# Finger joint names for right hand (29-DoF DEX3)
RIGHT_FINGER_JOINT_NAMES_29DOF = [
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]


def get_palm_forward(quat: torch.Tensor) -> torch.Tensor:
    """Extract forward direction (first column of rotation matrix) from wxyz quaternion."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y)
    ], dim=-1)


def compute_orientation_error(palm_quat: torch.Tensor, target_dir: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute angular error between palm forward and target direction."""
    forward = get_palm_forward(palm_quat)
    if target_dir is None:
        target_dir = torch.zeros_like(forward)
        target_dir[:, 2] = -1.0  # palm down
    dot = torch.clamp((forward * target_dir).sum(dim=-1), -1.0, 1.0)
    return torch.acos(dot)


class ArmPolicyWrapper:
    """
    Wrapper for the Stage 7 arm actor neural network.

    Loads weights from the DualActorCritic checkpoint and provides:
      - get_action(obs_55) -> action_5
      - build_obs(...) -> obs_55  (given robot state + target)

    Only controls RIGHT arm (5 joints). Left arm stays at default or heuristic.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Build arm actor: 55 -> [256, ELU, 256, ELU, 128, ELU] -> 5
        layers = []
        prev = ARM_OBS_DIM
        for h in [256, 256, 128]:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ARM_ACT_DIM))
        self.net = nn.Sequential(*layers).to(self.device)
        self.net.eval()

        # Load weights from checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))

        # Extract arm_actor weights
        arm_state = {}
        for key, val in state.items():
            if key.startswith("arm_actor.net."):
                new_key = key.replace("arm_actor.net.", "")
                arm_state[new_key] = val

        if not arm_state:
            raise RuntimeError(
                f"No arm_actor.net.* keys found in checkpoint! "
                f"Available keys: {[k for k in state.keys() if 'arm' in k]}"
            )

        self.net.load_state_dict(arm_state)

        # Load log_std for reference (not used in deterministic inference)
        self.log_std = state.get("arm_actor.log_std", torch.zeros(ARM_ACT_DIM))

        # Default arm pose (all zeros, as in 23-DoF training)
        self._default_arm = torch.tensor(
            ARM_DEFAULT, dtype=torch.float32, device=self.device
        )

        # Joint limits from 23-DoF training robot
        self._joint_lower = torch.tensor(
            ARM_JOINT_LOWER, dtype=torch.float32, device=self.device
        )
        self._joint_upper = torch.tensor(
            ARM_JOINT_UPPER, dtype=torch.float32, device=self.device
        )

        # Print info
        ckpt_name = os.path.basename(checkpoint_path)
        curriculum = ckpt.get("curriculum_level", "?")
        best_reward = ckpt.get("best_reward", "?")
        iteration = ckpt.get("iteration", ckpt.get("iter", "?"))
        print(f"[ArmPolicy Stage7] Architecture: {ARM_OBS_DIM} -> [256,256,128](ELU) -> {ARM_ACT_DIM}")
        print(f"[ArmPolicy Stage7] Checkpoint: {ckpt_name}")
        print(f"[ArmPolicy Stage7] iter={iteration}, reward={best_reward}, curriculum={curriculum}")
        print(f"[ArmPolicy Stage7] Action scale: {ARM_ACTION_SCALE}, default: all zeros")

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run arm actor inference.

        Args:
            obs: [N, 55] arm observation vector

        Returns:
            action: [N, 5] raw arm actions (before scaling)
        """
        return self.net(obs)

    def get_arm_targets(self, obs: torch.Tensor, smooth_alpha: float = 0.6) -> torch.Tensor:
        """
        Get absolute joint targets from arm policy.

        Args:
            obs: [N, 55] arm observation
            smooth_alpha: exponential smoothing factor (0=no smoothing, 1=full smoothing)

        Returns:
            targets: [N, 5] absolute joint positions for the 5 policy-controlled joints,
                     clamped to training joint limits and smoothed to prevent instability.
        """
        raw_action = self.get_action(obs)
        # target = default + action * scale  (matches Stage 7 training)
        targets = self._default_arm.unsqueeze(0) + raw_action * ARM_ACTION_SCALE
        # Clamp to conservative joint limits
        # Without this, the 29-DoF robot's wider limits allow the arm to
        # enter out-of-distribution states, causing a positive feedback loop
        # where the policy outputs increasingly extreme actions.
        targets = torch.max(targets, self._joint_lower.unsqueeze(0))
        targets = torch.min(targets, self._joint_upper.unsqueeze(0))
        # Exponential smoothing: blend with previous targets to reduce jitter
        if smooth_alpha > 0 and hasattr(self, '_prev_targets') and self._prev_targets is not None:
            targets = (1.0 - smooth_alpha) * targets + smooth_alpha * self._prev_targets
        self._prev_targets = targets.clone()
        return targets

    def reset_state(self, current_targets: "torch.Tensor | None" = None):
        """Reset internal state (call when activating the arm policy).

        Args:
            current_targets: [N, 5] current right arm joint positions.
                If provided, used as initial smoothing reference so the
                first step blends from the current pose instead of jumping.
        """
        if current_targets is not None:
            self._prev_targets = current_targets.clone()
        else:
            self._prev_targets = None

    @staticmethod
    def build_obs(
        arm_pos: torch.Tensor,           # [N, 5] right arm joint positions
        arm_vel: torch.Tensor,           # [N, 5] right arm joint velocities
        finger_pos: torch.Tensor,        # [N, 7] right finger joint positions
        ee_body: torch.Tensor,           # [N, 3] EE position in body frame
        ee_vel_body: torch.Tensor,       # [N, 3] EE velocity in body frame
        palm_quat: torch.Tensor,         # [N, 4] palm quaternion (wxyz)
        finger_lower: torch.Tensor,      # [7] finger lower limits
        finger_upper: torch.Tensor,      # [7] finger upper limits
        target_body: torch.Tensor,       # [N, 3] target position in body frame
        root_height: torch.Tensor,       # [N] current root z position
        lin_vel_body: torch.Tensor,      # [N, 3] body-frame linear velocity
        ang_vel_body: torch.Tensor,      # [N, 3] body-frame angular velocity
        steps_since_spawn: torch.Tensor, # [N] steps since arm was activated
        ee_pos_at_spawn: torch.Tensor,   # [N, 3] EE body pos when arm was activated
        initial_dist: torch.Tensor,      # [N] initial distance to target
        target_orient: Optional[torch.Tensor] = None,  # [N, 3] target orient dir
    ) -> torch.Tensor:
        """
        Build 55-dim observation matching Stage 7 training format exactly.

        Returns:
            obs: [N, 55] clamped to [-10, 10]
        """
        n = arm_pos.shape[0]
        device = arm_pos.device

        # Finger gripper ratio
        finger_range = finger_upper.unsqueeze(0) - finger_lower.unsqueeze(0) + 1e-6
        finger_normalized = (finger_pos - finger_lower.unsqueeze(0)) / finger_range
        gripper_closed_ratio = finger_normalized.mean(dim=-1, keepdim=True)

        # Grip force estimate
        grip_force = gripper_closed_ratio.clamp(0, 1)  # simplified

        # Contact detection
        pos_error = target_body - ee_body
        dist_to_target = pos_error.norm(dim=-1, keepdim=True)
        contact_detected = (dist_to_target < 0.08).float()

        # Position error normalized
        pos_dist = dist_to_target / 0.5

        # Orientation error
        if target_orient is None:
            target_orient = torch.zeros(n, 3, device=device)
            target_orient[:, 2] = -1.0  # palm down
        orient_err = compute_orientation_error(palm_quat, target_orient).unsqueeze(-1) / 3.14159

        # Target reached (curriculum level 7 thresholds)
        orient_threshold = 2.0  # Level 7
        target_reached = (
            (dist_to_target < OBS_POS_THRESHOLD) &
            (orient_err * 3.14159 < orient_threshold)
        ).float()

        # Height observations
        height_cmd = torch.full((n, 1), HEIGHT_DEFAULT, device=device)
        current_height = root_height.unsqueeze(-1) if root_height.ndim == 1 else root_height
        height_err = (height_cmd - current_height) / 0.4

        # Placeholders (always zero in inference)
        estimated_load = torch.zeros(n, 3, device=device)
        object_in_hand = torch.zeros(n, 1, device=device)

        # Velocity observations
        lin_vel_xy = lin_vel_body[:, :2]
        ang_vel_z = ang_vel_body[:, 2:3]

        # Anti-gaming observations
        max_steps = float(MAX_REACH_STEPS)
        steps_norm = (steps_since_spawn.float() / max_steps).unsqueeze(-1).clamp(0, 2)
        ee_displacement = (ee_body - ee_pos_at_spawn).norm(dim=-1, keepdim=True)
        initial_dist_obs = initial_dist.unsqueeze(-1) / 0.5

        # Assemble 55-dim observation (exact Stage 7 order)
        obs = torch.cat([
            arm_pos,                    # 5   [0:5]
            arm_vel * 0.1,              # 5   [5:10]
            finger_pos,                 # 7   [10:17]
            ee_body,                    # 3   [17:20]
            ee_vel_body,                # 3   [20:23]
            palm_quat,                  # 4   [23:27]
            grip_force,                 # 1   [27]
            gripper_closed_ratio,       # 1   [28]
            contact_detected,           # 1   [29]
            target_body,                # 3   [30:33]
            pos_error,                  # 3   [33:36]
            pos_dist,                   # 1   [36]
            orient_err,                 # 1   [37]
            target_reached,             # 1   [38]
            height_cmd,                 # 1   [39]
            current_height,             # 1   [40]
            height_err,                 # 1   [41]
            estimated_load,             # 3   [42:45]
            object_in_hand,             # 1   [45]
            target_orient,              # 3   [46:49]
            lin_vel_xy,                 # 2   [49:51]
            ang_vel_z,                  # 1   [51]
            steps_norm,                 # 1   [52]
            ee_displacement,            # 1   [53]
            initial_dist_obs,           # 1   [54]
        ], dim=-1)  # = 55

        return obs.clamp(-10, 10).nan_to_num()
