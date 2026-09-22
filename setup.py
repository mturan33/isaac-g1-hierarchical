"""Setup for high_low_hierarchical_g1 package."""

from setuptools import setup, find_packages

setup(
    name="high_low_hierarchical_g1",
    version="0.1.0",
    description="Hierarchical VLM+RL control for the Unitree G1 humanoid in Isaac Lab",
    url="https://github.com/mturan33/isaac-g1-hierarchical",
    author="Mehmet Turan Yardımcı",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
    extras_require={
        # only needed by the legacy text-only planner in legacy/planner/
        "llm": ["anthropic", "openai"],
    },
)
