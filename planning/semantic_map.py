"""
Semantic Map
=============
Dual-mode world state representation for the VLM planning pipeline.

Modes:
    - "ground_truth": Reads object positions directly from Isaac Lab sim
    - "perception": Stub for YOLO + depth camera pipeline (future)

Both modes produce identical JSON output via get_json(), so the VLM planner
and skill executor work without any changes regardless of mode.
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Any


class SemanticMap:
    """Dual-mode semantic map for robot world understanding.

    Args:
        mode: "ground_truth" (sim) or "perception" (camera-based)
        env: HierarchicalG1Env instance (required for ground_truth mode)
        perception_module: G1PerceptionModule instance (for perception mode)
    """

    # Arm workspace radius — objects within this distance are reachable
    ARM_REACH = 0.35  # meters

    def __init__(
        self,
        mode: str = "ground_truth",
        env: Any = None,
        perception_module: Any = None,
    ):
        self.mode = mode
        self.env = env
        self.perception = perception_module

        # World state
        self.objects: dict[str, dict] = {}
        self.surfaces: dict[str, dict] = {}
        self.robot_state: dict = {}

        # Validate
        if mode == "ground_truth" and env is None:
            raise ValueError("ground_truth mode requires env parameter")
        if mode == "perception" and perception_module is None:
            print("[SemanticMap] WARNING: perception mode but no perception module provided")

        print(f"[SemanticMap] Initialized in '{mode}' mode")

    def update(self, rgb=None, depth=None, camera_intrinsics=None):
        """Update world state. Call each frame before planning or execution."""
        if self.mode == "ground_truth":
            self._update_from_sim()
        else:
            self._update_from_perception(rgb, depth, camera_intrinsics)
        self._update_robot_state()

    # ------------------------------------------------------------------
    # Ground truth mode (Isaac Lab sim)
    # ------------------------------------------------------------------
    def _update_from_sim(self):
        """Read object/surface positions directly from simulation."""
        env = self.env

        # Cup
        cup_pos = env.cup.data.root_pos_w[0].cpu().tolist()
        robot_pos = env.robot.data.root_pos_w[0].cpu().tolist()
        cup_dist = math.sqrt(
            (cup_pos[0] - robot_pos[0]) ** 2
            + (cup_pos[1] - robot_pos[1]) ** 2
        )
        self.objects["cup_01"] = {
            "id": "cup_01",
            "class": "cup",
            "position_3d": cup_pos,
            "graspable": True,
            "distance_to_robot": round(cup_dist, 3),
            "reachable": cup_dist < self.ARM_REACH,
            "confidence": 1.0,
        }

        # Table 1 (front)
        table_pos = env.table.data.root_pos_w[0].cpu().tolist()
        self.surfaces["table_01"] = {
            "id": "table_01",
            "class": "table",
            "position_3d": table_pos,
            "placeable": True,
            "size": [0.8, 1.2, 0.75],
        }

        # Table 2 (behind robot)
        table2_pos = env.table2.data.root_pos_w[0].cpu().tolist()
        self.surfaces["table_02"] = {
            "id": "table_02",
            "class": "table",
            "position_3d": table2_pos,
            "placeable": True,
            "size": [0.8, 1.2, 0.75],
        }

    # ------------------------------------------------------------------
    # Perception mode (camera-based — stub)
    # ------------------------------------------------------------------
    def _update_from_perception(self, rgb, depth, intrinsics):
        """Update from YOLO + depth camera. Stub for future implementation."""
        if self.perception is None:
            print("[SemanticMap] Perception module not available, skipping update")
            return

        try:
            # Future: self.perception.detect(rgb, depth, intrinsics)
            # Parse Detection3D list into self.objects
            pass
        except Exception as e:
            print(f"[SemanticMap] Perception update failed: {e}")

    # ------------------------------------------------------------------
    # Robot state
    # ------------------------------------------------------------------
    def _update_robot_state(self):
        """Update robot position, heading, and holding status."""
        env = self.env
        if env is None:
            return

        root_pos = env.robot.data.root_pos_w[0].cpu().tolist()
        root_quat = env.robot.data.root_quat_w[0]

        # Heading from quaternion (yaw in degrees)
        # Uses same quat_to_euler_xyz_wxyz as hierarchical_env.py
        w, x, y, z = root_quat[0].item(), root_quat[1].item(), root_quat[2].item(), root_quat[3].item()
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        heading_deg = math.degrees(yaw_rad)

        # Holding detection: fingers closed + nearest object close
        holding = None
        if hasattr(env, 'finger_controller') and env.finger_controller.is_closed():
            # Check nearest graspable object distance
            for obj_id, obj_data in self.objects.items():
                if obj_data.get("graspable", False):
                    obj_pos = obj_data["position_3d"]
                    dist = math.sqrt(
                        (obj_pos[0] - root_pos[0]) ** 2
                        + (obj_pos[1] - root_pos[1]) ** 2
                    )
                    if dist < 0.5:  # close enough to be holding
                        holding = obj_id
                        break

        # Determine stance
        base_height = root_pos[2]
        if base_height < 0.5:
            stance = "squatting"
        else:
            stance = "standing"

        self.robot_state = {
            "position": root_pos,
            "heading_deg": round(heading_deg, 1),
            "holding": holding,
            "stance": stance,
        }

    # ------------------------------------------------------------------
    # JSON output (identical format in both modes)
    # ------------------------------------------------------------------
    def get_json(self) -> dict:
        """Return standardized world state for VLM planner.

        Output format is identical regardless of mode (GT or perception).
        """
        return {
            "robot": self.robot_state,
            "objects": list(self.objects.values()),
            "surfaces": list(self.surfaces.values()),
        }

    # ------------------------------------------------------------------
    # Position queries (for skill executor)
    # ------------------------------------------------------------------
    def get_object_position(self, object_id: str) -> Optional[list]:
        """Get real-time 3D position of an object.

        Args:
            object_id: Object identifier (e.g., "cup_01")

        Returns:
            [x, y, z] world position or None if not found
        """
        obj = self.objects.get(object_id)
        if obj is None:
            # Try matching by class name
            for oid, odata in self.objects.items():
                if odata["class"] in object_id or object_id in odata["class"]:
                    return odata["position_3d"]
            return None
        return obj["position_3d"]

    def get_surface_position(self, surface_id: str) -> Optional[list]:
        """Get real-time 3D position of a surface.

        Args:
            surface_id: Surface identifier (e.g., "table_01")

        Returns:
            [x, y, z] world position or None if not found
        """
        surf = self.surfaces.get(surface_id)
        if surf is None:
            # Try matching by class name or partial id
            for sid, sdata in self.surfaces.items():
                if sdata["class"] in surface_id or surface_id in sid:
                    return sdata["position_3d"]
            return None
        return surf["position_3d"]

    def get_position(self, target_id: str) -> Optional[list]:
        """Get position of any object or surface by id."""
        pos = self.get_object_position(target_id)
        if pos is not None:
            return pos
        return self.get_surface_position(target_id)

    def get_per_env_position(self, target_id: str) -> Optional[torch.Tensor]:
        """Get per-env world position tensor [num_envs, 3] from sim.

        In multi-env setups, each env has its own copy of objects at different
        world positions (due to env_spacing). This returns ALL envs' positions.

        Args:
            target_id: Object or surface identifier

        Returns:
            [num_envs, 3] tensor or None if target not found
        """
        if self.mode != "ground_truth" or self.env is None:
            return None

        # Map target_id to sim entity
        entity = self._resolve_entity(target_id)
        if entity is not None:
            return entity.data.root_pos_w.clone()  # [num_envs, 3]
        return None

    def _resolve_entity(self, target_id: str):
        """Resolve target_id to an Isaac Lab entity (RigidObject)."""
        env = self.env

        # Direct ID match
        id_map = {
            "cup_01": env.cup,
            "table_01": env.table,
            "table_02": env.table2,
        }
        if target_id in id_map:
            return id_map[target_id]

        # Class-based matching
        class_map = {
            "cup": env.cup,
            "table": env.table,
        }
        for class_name, entity in class_map.items():
            if class_name in target_id:
                # Check if it's the second table
                if "02" in target_id or "second" in target_id or "2" in target_id:
                    return env.table2
                return entity
        return None
