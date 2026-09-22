"""
Skill Executor
===============
Executes a plan (list of SkillSteps) by running skill primitives
in sequence, feeding velocity commands to the locomotion policy.

Handles:
  - Skill instantiation and parameter binding
  - Sequential execution with proper transitions
  - Failure detection and replanning (SayCan-inspired)
  - Semantic map updates between skills
"""

from __future__ import annotations

import torch
from typing import Optional

from .semantic_map import SemanticMap
from .llm_planner import LLMPlanner, SkillStep
from ...skills.base_skill import BaseSkill, SkillResult, SkillStatus
from ...skills.walk_to import WalkToSkill
from ...skills.turn_to import TurnToSkill
from ...skills.stand_still import StandStillSkill
from ...skills.squat import SquatSkill
from ...skills.heuristic_manipulation import HeuristicGraspSkill, HeuristicPlaceSkill
from ...low_level.policy_wrapper import LocomotionPolicy
from ...config.skill_config import SkillLibraryConfig, DEFAULT_SKILL_CONFIG


class SkillExecutor:
    """
    Executes a task plan using skill primitives.

    Architecture:
        LLM plan → SkillExecutor → Skill → velocity_cmd → LocoPolicy → joint_targets

    Usage:
        executor = SkillExecutor(loco_policy, semantic_map)
        result = executor.execute_plan(plan, obs_dict_fn, step_fn)
    """

    def __init__(
        self,
        loco_policy: LocomotionPolicy,
        semantic_map: SemanticMap,
        planner: Optional[LLMPlanner] = None,
        config: Optional[SkillLibraryConfig] = None,
        device: str = "cuda",
        max_replan_attempts: int = 3,
    ):
        self.loco_policy = loco_policy
        self.semantic_map = semantic_map
        self.planner = planner
        self.cfg = config or DEFAULT_SKILL_CONFIG
        self.device = device
        self.max_replan_attempts = max_replan_attempts

        # Initialize skill library
        self.skill_library: dict[str, BaseSkill] = {
            "walk_to": WalkToSkill(config=self.cfg.walk_to, device=device),
            "turn_to": TurnToSkill(config=self.cfg.turn_to, device=device),
            "stand_still": StandStillSkill(config=self.cfg.stand_still, device=device),
            "squat": SquatSkill(config=self.cfg.squat, device=device),
            "stand_up": StandStillSkill(config=self.cfg.stand_still, device=device),  # Alias
            "grasp": HeuristicGraspSkill(config=self.cfg.grasp, device=device),
            "place": HeuristicPlaceSkill(config=self.cfg.place, device=device),
        }

        # Execution history
        self.completed_steps: list[SkillStep] = []
        self.total_steps: int = 0

        print(f"[SkillExecutor] Initialized with {len(self.skill_library)} skills")

    def execute_plan(
        self,
        plan: list[SkillStep],
        get_obs_fn,
        step_env_fn,
        original_task: str = "",
    ) -> SkillResult:
        """
        Execute a full task plan.

        Args:
            plan: List of SkillSteps to execute.
            get_obs_fn: Callable that returns current obs_dict.
            step_env_fn: Callable(joint_targets) that steps the environment.
            original_task: Original task description (for replanning).

        Returns:
            Final SkillResult (success if all steps completed).
        """
        self.completed_steps = []
        self.total_steps = 0

        print(f"\n{'='*60}")
        print(f"[SkillExecutor] Executing plan with {len(plan)} steps")
        print(f"{'='*60}")

        for i, skill_step in enumerate(plan):
            print(f"\n--- Step {i+1}/{len(plan)}: {skill_step.skill}({skill_step.params}) ---")
            print(f"    Description: {skill_step.description}")

            # Get skill instance
            skill = self.skill_library.get(skill_step.skill)
            if skill is None:
                print(f"[SkillExecutor] Unknown skill: {skill_step.skill}")
                return SkillResult(
                    status=SkillStatus.FAILED,
                    steps_taken=self.total_steps,
                    reason=f"Unknown skill: {skill_step.skill}",
                )

            # Execute skill
            result = self._execute_skill(skill, skill_step.params, get_obs_fn, step_env_fn)

            if result.succeeded:
                self.completed_steps.append(skill_step)
                print(f"    Result: SUCCESS ({result.reason})")
            else:
                print(f"    Result: {result.status.value} ({result.reason})")

                # Attempt replanning
                if self.planner and original_task:
                    replan_result = self._try_replan(
                        original_task, skill_step, result,
                        get_obs_fn, step_env_fn,
                    )
                    if replan_result and replan_result.succeeded:
                        continue
                    return replan_result or result
                else:
                    return result

        # All steps completed
        print(f"\n{'='*60}")
        print(f"[SkillExecutor] Plan completed! Total steps: {self.total_steps}")
        print(f"{'='*60}")

        return SkillResult(
            status=SkillStatus.SUCCESS,
            steps_taken=self.total_steps,
            reason="All plan steps completed",
        )

    def _execute_skill(
        self,
        skill: BaseSkill,
        params: dict,
        get_obs_fn,
        step_env_fn,
    ) -> SkillResult:
        """Execute a single skill until completion."""
        # Initialize skill with parameters
        skill.reset(**params)

        while True:
            # Get current observations
            obs_dict = get_obs_fn()

            # Skill step: get velocity command
            cmd_vel, done, result = skill.step(obs_dict)

            if done:
                # Run one final zero-command step for stability
                zero_cmd = torch.zeros(1, 3, device=self.device)
                joint_targets = self.loco_policy.get_action(
                    base_ang_vel=obs_dict["base_ang_vel"],
                    projected_gravity=obs_dict["projected_gravity"],
                    joint_pos=obs_dict["joint_pos"],
                    joint_vel=obs_dict["joint_vel"],
                    velocity_command=zero_cmd,
                )
                step_env_fn(joint_targets)
                self.total_steps += 1
                return result

            # Convert velocity command to joint targets via locomotion policy
            joint_targets = self.loco_policy.get_action(
                base_ang_vel=obs_dict["base_ang_vel"],
                projected_gravity=obs_dict["projected_gravity"],
                joint_pos=obs_dict["joint_pos"],
                joint_vel=obs_dict["joint_vel"],
                velocity_command=cmd_vel,
            )

            # Step the environment
            step_env_fn(joint_targets)
            self.total_steps += 1

            # Update semantic map periodically
            if self.total_steps % 50 == 0:
                self.semantic_map.update_robot(
                    root_pos=obs_dict["root_pos"][0],
                    root_quat=obs_dict["root_quat"][0],
                )

    def _try_replan(
        self,
        original_task: str,
        failed_step: SkillStep,
        failure_result: SkillResult,
        get_obs_fn,
        step_env_fn,
    ) -> Optional[SkillResult]:
        """Attempt replanning after a failure."""
        for attempt in range(self.max_replan_attempts):
            print(f"\n[SkillExecutor] Replan attempt {attempt + 1}/{self.max_replan_attempts}")

            # Update semantic map
            obs = get_obs_fn()
            self.semantic_map.update_robot(
                root_pos=obs["root_pos"][0],
                root_quat=obs["root_quat"][0],
            )

            # Ask LLM for new plan
            new_plan = self.planner.replan(
                original_task=original_task,
                completed_steps=self.completed_steps,
                failure_reason=failure_result.reason,
                semantic_map_state=self.semantic_map.get_state(),
            )

            if not new_plan:
                print("[SkillExecutor] Replan returned empty plan")
                continue

            # Execute new plan
            result = self.execute_plan(
                new_plan, get_obs_fn, step_env_fn,
                original_task=original_task,
            )

            if result.succeeded:
                return result

        print("[SkillExecutor] All replan attempts exhausted")
        return None

    def execute_single_skill(
        self,
        skill_name: str,
        params: dict,
        get_obs_fn,
        step_env_fn,
    ) -> SkillResult:
        """
        Execute a single skill (for testing).

        Args:
            skill_name: Name of the skill to execute.
            params: Skill parameters.
            get_obs_fn: Callable returning obs_dict.
            step_env_fn: Callable(joint_targets) stepping the env.

        Returns:
            SkillResult
        """
        skill = self.skill_library.get(skill_name)
        if skill is None:
            return SkillResult(
                status=SkillStatus.FAILED,
                reason=f"Unknown skill: {skill_name}",
            )

        return self._execute_skill(skill, params, get_obs_fn, step_env_fn)
