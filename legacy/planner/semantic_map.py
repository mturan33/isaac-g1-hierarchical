"""
Semantic Map
=============
Ground-truth world state extracted from Isaac Lab simulation.

Provides the LLM planner with structured information about:
  - Robot state (position, orientation, stance, holding)
  - Object positions and properties
  - Surface positions and properties

In future phases, this can be replaced with vision-based perception
(using the VLM wrapper from old/vlm_wrapper.py).
"""

from __future__ import annotations

import json
import math
import torch
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class ObjectInfo:
    """Information about a scene object."""
    id: str
    type: str                          # "cup", "box", "ball", etc.
    position: list[float]              # [x, y, z] world frame
    graspable: bool = True
    color: str = "unknown"
    on_surface: Optional[str] = None   # ID of surface it's on, or None (ground)


@dataclass
class SurfaceInfo:
    """Information about a surface (table, shelf, etc.)."""
    id: str
    type: str                          # "table", "shelf", "counter"
    position: list[float]              # [x, y, z] center
    size: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.75])
    placeable: bool = True


@dataclass
class RobotState:
    """Current robot state for the planner."""
    position: list[float]              # [x, y, z]
    orientation: list[float]           # [qw, qx, qy, qz]
    heading: float                     # yaw in radians
    stance: str = "standing"           # "standing" | "squatting"
    holding: Optional[str] = None      # object_id or None
    base_height: float = 0.78


class SemanticMap:
    """
    Ground-truth semantic map from Isaac Lab simulation.

    Extracts robot state, object positions, and surface info
    directly from the simulation environment.
    """

    def __init__(self):
        self.robot = RobotState(
            position=[0.0, 0.0, 0.8],
            orientation=[1.0, 0.0, 0.0, 0.0],
            heading=0.0,
        )
        self.objects: dict[str, ObjectInfo] = {}
        self.surfaces: dict[str, SurfaceInfo] = {}

    # ================================================================
    # Setup — register objects and surfaces in the scene
    # ================================================================

    def add_object(
        self,
        id: str,
        type: str,
        position: list[float],
        graspable: bool = True,
        color: str = "unknown",
    ) -> None:
        """Register a scene object."""
        self.objects[id] = ObjectInfo(
            id=id, type=type, position=position,
            graspable=graspable, color=color,
        )

    def add_surface(
        self,
        id: str,
        type: str,
        position: list[float],
        size: list[float] = None,
    ) -> None:
        """Register a surface."""
        self.surfaces[id] = SurfaceInfo(
            id=id, type=type, position=position,
            size=size or [1.0, 0.5, 0.75],
        )

    # ================================================================
    # Update — called each step from environment
    # ================================================================

    def update_robot(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        base_height: float = None,
        holding: str = None,
    ) -> None:
        """
        Update robot state from simulation.

        Args:
            root_pos: [3] tensor — robot base position
            root_quat: [4] tensor — quaternion [w, x, y, z]
            base_height: Optional override for height
            holding: ID of held object, or None
        """
        pos = root_pos.cpu().tolist() if torch.is_tensor(root_pos) else list(root_pos)
        quat = root_quat.cpu().tolist() if torch.is_tensor(root_quat) else list(root_quat)

        # Compute yaw from quaternion
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        height = base_height if base_height is not None else pos[2]

        self.robot = RobotState(
            position=pos,
            orientation=quat,
            heading=yaw,
            stance="squatting" if height < 0.5 else "standing",
            holding=holding,
            base_height=height,
        )

    def update_object(self, id: str, position: torch.Tensor) -> None:
        """Update an object's position from simulation."""
        if id in self.objects:
            pos = position.cpu().tolist() if torch.is_tensor(position) else list(position)
            self.objects[id].position = pos

    # ================================================================
    # Query — used by planner and skills
    # ================================================================

    def get_state(self) -> dict[str, Any]:
        """
        Get full state as JSON-serializable dict.
        This is what gets sent to the LLM planner.
        """
        return {
            "robot": {
                "position": self.robot.position,
                "orientation": self.robot.orientation,
                "heading_deg": round(math.degrees(self.robot.heading), 1),
                "stance": self.robot.stance,
                "holding": self.robot.holding,
                "base_height": round(self.robot.base_height, 3),
            },
            "objects": [
                {
                    "id": obj.id,
                    "type": obj.type,
                    "position": [round(p, 3) for p in obj.position],
                    "graspable": obj.graspable,
                    "color": obj.color,
                    "on_surface": obj.on_surface,
                }
                for obj in self.objects.values()
            ],
            "surfaces": [
                {
                    "id": surf.id,
                    "type": surf.type,
                    "position": [round(p, 3) for p in surf.position],
                    "placeable": surf.placeable,
                }
                for surf in self.surfaces.values()
            ],
        }

    def get_state_json(self, indent: int = 2) -> str:
        """Get state as formatted JSON string."""
        return json.dumps(self.get_state(), indent=indent, ensure_ascii=False)

    def get_object_position(self, object_id: str) -> Optional[list[float]]:
        """Get an object's position by ID."""
        obj = self.objects.get(object_id)
        return obj.position if obj else None

    def get_surface_position(self, surface_id: str) -> Optional[list[float]]:
        """Get a surface's position by ID."""
        surf = self.surfaces.get(surface_id)
        return surf.position if surf else None

    def get_distance_to(self, target_id: str) -> Optional[float]:
        """Get robot's distance to an object or surface."""
        pos = self.get_object_position(target_id) or self.get_surface_position(target_id)
        if pos is None:
            return None
        dx = self.robot.position[0] - pos[0]
        dy = self.robot.position[1] - pos[1]
        return (dx**2 + dy**2) ** 0.5

    def __repr__(self) -> str:
        return (
            f"SemanticMap("
            f"robot=({self.robot.position[0]:.1f}, {self.robot.position[1]:.1f}), "
            f"objects={len(self.objects)}, surfaces={len(self.surfaces)})"
        )
