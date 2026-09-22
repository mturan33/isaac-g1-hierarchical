"""
LLM Task Planner
==================
Decomposes natural language task descriptions into executable skill sequences.

Inspired by:
  - SayCan (Ahn et al. 2022): LLM × affordance scoring
  - Berkeley Loco-Manipulation (Ouyang et al. 2024): 3-stage LLM cascade

Supports Claude API (Anthropic) and GPT-4 API (OpenAI).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class SkillStep:
    """A single step in the task plan."""
    skill: str                              # Skill name: walk_to, turn_to, etc.
    params: dict[str, Any] = field(default_factory=dict)
    termination: str = ""                   # Human-readable termination condition
    description: str = ""                   # What this step does


SYSTEM_PROMPT = """Sen bir robot task planner'sın. Unitree G1 humanoid robotun aşağıdaki skill'leri var:

SKILLS:
- walk_to(target_x, target_y): Robot (x,y) konumuna yürür. Parametreler metre cinsinden.
- turn_to(heading): Robot belirtilen yöne döner (radyan).
- turn_to(target_x, target_y): Robot belirtilen konuma doğru döner.
- stand_still(duration_s): Robot belirtilen süre durur.
- squat(depth): Robot çömelir (depth: 0.0-0.4 metre).
- stand_up(): Robot ayağa kalkar (squat'tan sonra).
- grasp(object_id): Nesneyi tutar (el yakınında olmalı).
- place(surface_id): Tuttuğu nesneyi yüzeye bırakır.

KURALLAR:
1. grasp() öncesi robot nesneye yakın olmalı (walk_to ile yaklaş).
2. Yerdeki nesneler için: walk_to → squat → grasp → stand_up sırası.
3. Masadaki nesneler için: walk_to → grasp yeterli (squat gerekmez).
4. place() öncesi robot yüzeye yakın olmalı.
5. Bir seferde tek nesne tutulabilir.
6. walk_to hedefi nesnenin biraz yakınına olmalı (tam üzerine değil, 0.3m öteye).
7. stand_still, skill'ler arasında stabilizasyon için kullanılır.

ÇIKTI FORMATI:
JSON array döndür. Her eleman: {"skill": "...", "params": {...}, "description": "..."}

ÖNEMLİ: Sadece JSON array döndür, başka metin ekleme."""

SYSTEM_PROMPT_EN = """You are a robot task planner. The Unitree G1 humanoid robot has these skills:

SKILLS:
- walk_to(target_x, target_y): Walk to (x,y) position. Parameters in meters.
- turn_to(heading): Turn to specified heading (radians).
- turn_to(target_x, target_y): Turn to face a position.
- stand_still(duration_s): Stand still for duration.
- squat(depth): Squat down (depth: 0.0-0.4 meters).
- stand_up(): Stand up from squat.
- grasp(object_id): Grasp an object (hand must be near it).
- place(surface_id): Place held object on surface.

RULES:
1. Before grasp(), robot must be near the object (use walk_to).
2. For ground objects: walk_to → squat → grasp → stand_up sequence.
3. For table objects: walk_to → grasp is enough (no squat needed).
4. Before place(), robot must be near the surface.
5. Only one object can be held at a time.
6. walk_to target should be slightly before the object (0.3m offset).
7. stand_still is used for stabilization between skills.

OUTPUT FORMAT:
Return a JSON array. Each element: {"skill": "...", "params": {...}, "description": "..."}

IMPORTANT: Return ONLY the JSON array, no other text."""


class LLMPlanner:
    """
    LLM-based task planner.

    Decomposes natural language tasks into skill sequences.
    """

    def __init__(
        self,
        provider: str = "anthropic",      # "anthropic" or "openai"
        model: str = None,
        api_key: str = None,
        language: str = "tr",              # "tr" for Turkish, "en" for English
        temperature: float = 0.1,          # Low for deterministic plans
    ):
        self.provider = provider
        self.language = language
        self.temperature = temperature

        if provider == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "openai":
            self.model = model or "gpt-4"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._system_prompt = SYSTEM_PROMPT if language == "tr" else SYSTEM_PROMPT_EN
        self._client = None

        print(f"[LLMPlanner] Provider: {provider}, Model: {self.model}")

    def _get_client(self):
        """Lazy-initialize API client."""
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def plan(
        self,
        task_description: str,
        semantic_map_state: dict,
    ) -> list[SkillStep]:
        """
        Generate a task plan from natural language description.

        Args:
            task_description: Natural language task (e.g., "Bardağı al, masaya koy")
            semantic_map_state: Current world state from SemanticMap.get_state()

        Returns:
            List of SkillStep objects representing the plan.
        """
        user_message = (
            f"Task: {task_description}\n\n"
            f"Current State:\n{json.dumps(semantic_map_state, indent=2, ensure_ascii=False)}"
        )

        print(f"[LLMPlanner] Planning: '{task_description}'")

        try:
            raw_plan = self._call_llm(user_message)
            steps = self._parse_plan(raw_plan)
            print(f"[LLMPlanner] Generated {len(steps)} steps:")
            for i, step in enumerate(steps):
                print(f"  {i+1}. {step.skill}({step.params}) — {step.description}")
            return steps

        except Exception as e:
            print(f"[LLMPlanner] Error: {e}")
            return []

    def replan(
        self,
        original_task: str,
        completed_steps: list[SkillStep],
        failure_reason: str,
        semantic_map_state: dict,
    ) -> list[SkillStep]:
        """
        Generate a recovery plan after a skill failure (SayCan-inspired).

        Args:
            original_task: The original task description.
            completed_steps: Steps that were successfully completed.
            failure_reason: Why the current step failed.
            semantic_map_state: Current world state.

        Returns:
            New list of SkillStep objects for the remaining task.
        """
        completed_desc = "\n".join(
            f"  {i+1}. {s.skill}({s.params}) — DONE"
            for i, s in enumerate(completed_steps)
        )

        user_message = (
            f"Original Task: {original_task}\n\n"
            f"Completed Steps:\n{completed_desc}\n\n"
            f"FAILURE: {failure_reason}\n\n"
            f"Current State:\n{json.dumps(semantic_map_state, indent=2, ensure_ascii=False)}\n\n"
            f"Generate a NEW plan to complete the remaining task from the current state."
        )

        print(f"[LLMPlanner] Replanning due to: {failure_reason}")
        try:
            raw_plan = self._call_llm(user_message)
            return self._parse_plan(raw_plan)
        except Exception as e:
            print(f"[LLMPlanner] Replan error: {e}")
            return []

    def _call_llm(self, user_message: str) -> str:
        """Call the LLM API and return the response text."""
        client = self._get_client()

        if self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                system=self._system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        elif self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=2000,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content

    def _parse_plan(self, raw_text: str) -> list[SkillStep]:
        """Parse LLM output into SkillStep objects."""
        # Try to extract JSON from the response
        text = raw_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Parse JSON array
        plan_data = json.loads(text)

        steps = []
        for item in plan_data:
            steps.append(SkillStep(
                skill=item["skill"],
                params=item.get("params", {}),
                description=item.get("description", ""),
                termination=item.get("termination", ""),
            ))

        return steps

    def plan_offline(
        self,
        task_description: str,
        semantic_map_state: dict,
    ) -> list[SkillStep]:
        """
        Generate plan without LLM API (rule-based fallback).
        Useful for testing without API key.
        """
        print(f"[LLMPlanner] OFFLINE mode - rule-based plan for: '{task_description}'")

        # Simple pick-place pattern
        task_lower = task_description.lower()

        is_pick_place = (
            ("al" in task_lower and "koy" in task_lower) or  # Turkish
            ("pick" in task_lower and ("place" in task_lower or "put" in task_lower))  # English
        )
        if is_pick_place:
            # Find a graspable object and a placeable surface
            objects = semantic_map_state.get("objects", [])
            surfaces = semantic_map_state.get("surfaces", [])

            if objects and surfaces:
                obj = objects[0]
                surf = surfaces[0]
                obj_pos = obj["position"]
                surf_pos = surf["position"]

                return [
                    SkillStep("walk_to", {"target_x": obj_pos[0] - 0.3, "target_y": obj_pos[1]},
                              description=f"Walk near {obj['type']}"),
                    SkillStep("squat", {"depth": 0.3},
                              description="Squat to reach object"),
                    SkillStep("grasp", {"object_id": obj["id"]},
                              description=f"Grasp {obj['type']}"),
                    SkillStep("stand_still", {"duration_s": 1.0},
                              description="Stabilize"),
                    SkillStep("walk_to", {"target_x": surf_pos[0] - 0.3, "target_y": surf_pos[1]},
                              description=f"Walk to {surf['type']}"),
                    SkillStep("place", {"surface_id": surf["id"]},
                              description=f"Place on {surf['type']}"),
                ]

        # Default: just walk forward
        return [
            SkillStep("walk_to", {"target_x": 2.0, "target_y": 0.0},
                      description="Walk forward 2m"),
            SkillStep("stand_still", {"duration_s": 2.0},
                      description="Stop and stabilize"),
        ]
