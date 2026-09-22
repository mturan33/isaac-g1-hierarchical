"""
Skill Testing Script
=====================
Test individual skill primitives without the full hierarchical pipeline.
Useful for validating that the locomotion policy + skill wrappers work.

Usage:
    cd C:\\IsaacLab
    python source/isaaclab_tasks/isaaclab_tasks/direct/high_low_hierarchical_g1/scripts/test_skills.py \\
        --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/<run>/model_1500.pt \\
        --skill walk_to --target_x 3.0 --target_y 0.0

    # Test stand still:
    python test_skills.py --checkpoint <path> --skill stand_still --duration 5.0

    # Test turn to:
    python test_skills.py --checkpoint <path> --skill turn_to --heading 1.57
"""

import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Test G1 Skill Primitives")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Locomotion checkpoint for simulation tests (not implemented yet; "
                             "the offline tests below do not need one)")
    parser.add_argument("--skill", type=str, default="walk_to",
                        choices=["walk_to", "turn_to", "stand_still", "all"],
                        help="Skill to test")
    parser.add_argument("--device", type=str, default="cuda")

    # walk_to params
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=0.0)

    # turn_to params
    parser.add_argument("--heading", type=float, default=1.57,
                        help="Target heading in radians")

    # stand_still params
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Stand still duration in seconds")

    return parser.parse_args()


def test_offline():
    """
    Test skills without Isaac Lab simulation (structure validation only).
    Uses mock observation data.
    """
    import torch
    # skills/ uses package-relative imports (..low_level), so import through the package.
    sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
    from high_low_hierarchical_g1.config.joint_config import NUM_ALL_JOINTS
    from high_low_hierarchical_g1.skills.base_skill import SkillStatus
    from high_low_hierarchical_g1.skills.walk_to import WalkToSkill
    from high_low_hierarchical_g1.skills.turn_to import TurnToSkill
    from high_low_hierarchical_g1.skills.stand_still import StandStillSkill

    device = "cpu"  # No GPU needed for structure test
    print("\n" + "=" * 60)
    print("  Offline Skill Structure Test (no simulation)")
    print("=" * 60)

    # Mock observation dict
    mock_obs = {
        "root_pos": torch.tensor([[0.0, 0.0, 0.78]]),
        "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),  # Identity
        "base_ang_vel": torch.zeros(1, 3),
        "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
        "joint_pos": torch.zeros(1, NUM_ALL_JOINTS),
        "joint_vel": torch.zeros(1, NUM_ALL_JOINTS),
        "base_height": torch.tensor([0.78]),
    }

    # Test WalkTo
    print("\n--- Testing WalkToSkill ---")
    walk = WalkToSkill(device=device)
    walk.reset(target_x=3.0, target_y=0.0)

    cmd, done, result = walk.step(mock_obs)
    print(f"  cmd_vel: [{cmd[0,0]:.3f}, {cmd[0,1]:.3f}, {cmd[0,2]:.3f}]")
    print(f"  done: {done}, status: {result.status.value}")
    assert not done, "Should not be done on first step (3m away)"
    assert cmd[0, 0] > 0, "Should have positive vx (walking forward)"

    # Test TurnTo
    print("\n--- Testing TurnToSkill ---")
    turn = TurnToSkill(device=device)
    turn.reset(heading=1.57)

    cmd, done, result = turn.step(mock_obs)
    print(f"  cmd_vel: [{cmd[0,0]:.3f}, {cmd[0,1]:.3f}, {cmd[0,2]:.3f}]")
    print(f"  done: {done}, status: {result.status.value}")
    assert cmd[0, 0] == 0, "vx should be 0 for turn-in-place"
    assert cmd[0, 2] != 0, "vyaw should be non-zero"

    # Test StandStill
    print("\n--- Testing StandStillSkill ---")
    stand = StandStillSkill(device=device)
    stand.reset(duration_s=2.0)

    cmd, done, result = stand.step(mock_obs)
    print(f"  cmd_vel: [{cmd[0,0]:.3f}, {cmd[0,1]:.3f}, {cmd[0,2]:.3f}]")
    print(f"  done: {done}, status: {result.status.value}")
    assert (cmd == 0).all(), "All velocities should be zero"
    assert not done, "Should not be done on first step"

    # Test affordance
    print("\n--- Testing Affordances ---")
    state = {
        "robot": {"position": [0, 0, 0.78], "stance": "standing", "holding": None}
    }
    print(f"  walk_to affordance: {walk.get_affordance(state):.2f}")
    print(f"  turn_to affordance: {turn.get_affordance(state):.2f}")
    print(f"  stand_still affordance: {stand.get_affordance(state):.2f}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)


def test_planner_offline():
    """Test LLM planner in offline mode (no API key needed)."""
    # The legacy planner lives inside the package and uses package-relative imports,
    # so it is imported by its package path with the package parent on sys.path.
    sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
    from high_low_hierarchical_g1.legacy.planner.semantic_map import SemanticMap
    from high_low_hierarchical_g1.legacy.planner.llm_planner import LLMPlanner

    print("\n" + "=" * 60)
    print("  Offline Planner Test")
    print("=" * 60)

    # Setup semantic map
    sem_map = SemanticMap()
    sem_map.add_object("cup_01", "cup", [2.0, 1.0, 0.0], color="red")
    sem_map.add_surface("table_01", "table", [4.0, 0.0, 0.75])

    print(f"\nSemantic Map:\n{sem_map.get_state_json()}")

    # Plan offline
    planner = LLMPlanner(provider="anthropic")  # Won't call API in offline mode
    plan = planner.plan_offline(
        "Pick up the cup, place it on the table",
        sem_map.get_state(),
    )

    print(f"\nGenerated plan ({len(plan)} steps):")
    for i, step in enumerate(plan):
        print(f"  {i+1}. {step.skill}({step.params}) - {step.description}")

    assert len(plan) > 0, "Plan should not be empty"
    assert plan[0].skill == "walk_to", "First step should be walk_to"

    print("\n  PLANNER TEST PASSED!")


def main():
    args = parse_args()

    # Always run offline tests first (no dependencies)
    test_offline()
    test_planner_offline()

    # If checkpoint is provided and exists, run simulation tests
    if args.checkpoint and os.path.exists(args.checkpoint):
        print("\n" + "=" * 60)
        print("  Simulation Test (requires Isaac Lab)")
        print("=" * 60)
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Skill: {args.skill}")
        print(f"  NOTE: Simulation tests require Isaac Lab runtime.")
        print(f"  Run with: isaaclab.bat -p test_skills.py --checkpoint <path>")
        # TODO: Implement simulation-based testing
    else:
        print("\n  No checkpoint given (or not found); skipping simulation tests.")
        print("  Shipped policies are in checkpoints/ -- see the README.")


if __name__ == "__main__":
    main()
