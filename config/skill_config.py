"""
Skill Configuration
====================
Parameters, thresholds, and tuning constants for skill primitives.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WalkToConfig:
    """Configuration for walk_to skill."""
    # Arrival threshold
    position_threshold: float = 0.3        # meters — stop when within this distance
    heading_threshold: float = 0.15        # radians — acceptable heading error
    stop_distance: float = 0.0             # meters — stop THIS far from target (0=use position_threshold)

    # Velocity control
    max_forward_vel: float = 0.8           # m/s — maximum forward speed
    max_lateral_vel: float = 0.25          # m/s — maximum lateral speed
    max_yaw_rate: float = 0.2             # rad/s — maximum turn rate
    min_forward_vel: float = 0.1           # m/s — minimum (to avoid creeping)

    # P-controller gains for velocity command generation
    kp_linear: float = 1.0                 # Proportional gain for linear velocity
    kp_angular: float = 0.8               # Proportional gain for angular velocity

    # Heading alignment before walking
    align_first: bool = True               # Turn to face target before walking
    align_heading_threshold: float = 0.3   # rad — heading error to start walking

    # Safety
    max_steps: int = 2000                  # ~40 seconds at 50 Hz
    timeout_s: float = 40.0                # Episode timeout


@dataclass
class TurnToConfig:
    """Configuration for turn_to skill."""
    heading_threshold: float = 0.1         # radians
    max_yaw_rate: float = 0.2             # rad/s
    kp_angular: float = 0.5               # P gain
    max_steps: int = 500                   # ~10 seconds
    timeout_s: float = 10.0


@dataclass
class StandStillConfig:
    """Configuration for stand_still skill."""
    duration_s: float = 3.0                # Default standing duration
    max_drift: float = 0.5                # meters — reposition if drifted too far
    max_steps: int = 500                   # ~10 seconds


@dataclass
class SquatConfig:
    """Configuration for squat skill (placeholder for Phase 2)."""
    target_height: float = 0.45            # meters — squat depth
    height_threshold: float = 0.05         # meters — acceptable height error
    max_steps: int = 500
    timeout_s: float = 10.0


@dataclass
class HeuristicGraspConfig:
    """Configuration for heuristic grasp (Phase 3)."""
    grasp_distance: float = 0.05           # meters — max hand-object distance for grasp
    approach_distance: float = 0.3         # meters — distance to start approach
    max_steps: int = 200
    timeout_s: float = 4.0


@dataclass
class HeuristicPlaceConfig:
    """Configuration for heuristic place (Phase 3)."""
    place_height: float = 0.05             # meters — height above surface to release
    max_steps: int = 200
    timeout_s: float = 4.0


@dataclass
class SkillLibraryConfig:
    """Master config for all skills."""
    walk_to: WalkToConfig = field(default_factory=WalkToConfig)
    turn_to: TurnToConfig = field(default_factory=TurnToConfig)
    stand_still: StandStillConfig = field(default_factory=StandStillConfig)
    squat: SquatConfig = field(default_factory=SquatConfig)
    grasp: HeuristicGraspConfig = field(default_factory=HeuristicGraspConfig)
    place: HeuristicPlaceConfig = field(default_factory=HeuristicPlaceConfig)


# Default configuration instance
DEFAULT_SKILL_CONFIG = SkillLibraryConfig()
