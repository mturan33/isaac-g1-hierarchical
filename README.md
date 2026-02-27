# Hierarchical VLM+RL Control for G1 Humanoid

VLM task planner + RL skill primitives for the Unitree G1 (29 DoF) in Isaac Lab.

## Architecture

```
User Command (NL)  →  VLM Planner  →  Skill Executor  →  Loco Policy  →  G1 Robot
                         ↑                                    ↑
                    Semantic Map                        unitree_rl_lab
                    (ground truth)                      (29 DoF velocity)
```

## Quick Start

### 1. Install unitree_rl_lab
```bash
cd C:\IsaacLab\source\unitree_rl_lab\source\unitree_rl_lab
pip install -e .
```

### 2. Download G1 robot model
```bash
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
# Set UNITREE_MODEL_DIR in unitree_rl_lab/assets/robots/unitree.py
```

### 3. Train locomotion policy
```bash
cd C:\IsaacLab
.\isaaclab.bat -p source\unitree_rl_lab\scripts\rsl_rl\train.py ^
    --headless --task Unitree-G1-29dof-Velocity --num_envs 4096
```

### 4. Run tests
```bash
python scripts/test_skills.py --checkpoint <path_to_checkpoint>
```

## Project Structure

```
config/          Joint configs, skill parameters
low_level/       Locomotion policy wrapper
skills/          Skill primitives (walk_to, turn_to, stand_still, squat, grasp, place)
planner/         VLM planner, semantic map, skill executor
envs/            Isaac Lab environments
scripts/         Training, testing, demo scripts
```

## Skills

| Skill | Status | Description |
|-------|--------|-------------|
| walk_to | Phase 1 | Navigate to (x,y) position |
| turn_to | Phase 1 | Rotate to target heading |
| stand_still | Phase 1 | Hold position |
| squat | Phase 2 | Squat to target height (placeholder) |
| grasp | Phase 3 | Heuristic grasp (placeholder) |
| place | Phase 3 | Heuristic place (placeholder) |

## References

- SayCan (Ahn et al. 2022) - VLM + affordance scoring
- Berkeley Loco-Manipulation (Ouyang et al. 2024) - Skill chaining + VLM cascade
- unitree_rl_lab - G1 29-DoF locomotion training
