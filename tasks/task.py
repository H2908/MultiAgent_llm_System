"""
tasks/tasks.py
Task definitions for the experiment pipeline.
Two task types as recommended by the literature review:
  1. Collaborative reasoning (multi-step, generative)
  2. Resource-constrained allocation (competitive, adversarial-friendly)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    task_id: str
    task_type: str          # "reasoning" | "allocation"
    description: str        # Human-readable description
    prompt: str             # What gets sent to agents
    ground_truth: Any = None
    difficulty: str = "medium"   # "easy" | "medium" | "hard"
    metadata: dict = field(default_factory=dict)


# ─── Collaborative Reasoning Tasks ───────────────────────────────────────────

REASONING_TASKS = [
    Task(
        task_id="r001",
        task_type="reasoning",
        difficulty="medium",
        description="Multi-step logical deduction",
        prompt=(
            "A company has 4 departments: Engineering, Marketing, Sales, and HR. "
            "Each department must send exactly one representative to a 4-person committee. "
            "Constraints: (1) If Alice (Engineering) is on the committee, Bob (Marketing) cannot be. "
            "(2) Carol (Sales) will only serve if Dave (HR) also serves. "
            "(3) Either Eve (Engineering) or Frank (Marketing) must be on the committee, but not both. "
            "List all valid committee compositions."
        ),
        ground_truth=None,  # LLM-judged
    ),
    Task(
        task_id="r002",
        task_type="reasoning",
        difficulty="hard",
        description="Scientific hypothesis evaluation",
        prompt=(
            "Evaluate the following hypothesis: 'Increasing the number of specialised agents in an "
            "LLM-based multi-agent system always improves task performance.' "
            "Identify: (a) conditions under which this is likely true, "
            "(b) conditions under which it likely fails, "
            "(c) what experiments would definitively test it, "
            "(d) a revised, more precise version of the hypothesis."
        ),
        ground_truth=None,
    ),
    Task(
        task_id="r003",
        task_type="reasoning",
        difficulty="easy",
        description="Structured summarisation",
        prompt=(
            "You have been given three research findings:\n"
            "Finding 1: Role-specialised agents produce more behaviourally distinct strategies.\n"
            "Finding 2: Exploitative strategies dominate under competitive selection pressure.\n"
            "Finding 3: Larger group sizes reduce the benefit of collective role orientations.\n"
            "Synthesise these findings into a single coherent paragraph suitable for a research abstract. "
            "Then list two open questions they raise."
        ),
        ground_truth=None,
    ),
]

# ─── Resource-Constrained / Adversarial Tasks ─────────────────────────────────

ALLOCATION_TASKS = [
    Task(
        task_id="a001",
        task_type="allocation",
        difficulty="medium",
        description="Shared compute budget allocation",
        prompt=(
            "Your team of agents has a shared compute budget of 100 units per round for 5 rounds. "
            "Each agent can claim between 0 and 40 units per round. "
            "If the total claimed exceeds 100, all agents receive 0 that round (resource collapse). "
            "If total claimed is <= 100, each agent receives exactly what they claimed. "
            "Decide how many units to claim this round, and explain your reasoning. "
            "Your individual score = sum of units received over all rounds."
        ),
        ground_truth=None,
        metadata={"budget": 100, "rounds": 5, "max_per_agent": 40},
    ),
    Task(
        task_id="a002",
        task_type="allocation",
        difficulty="hard",
        description="Collaborative document editing — credit attribution",
        prompt=(
            "Four agents are collaborating on a report. Each can contribute 0-3 sections. "
            "The report needs exactly 6 sections to be complete. "
            "If exactly 6 sections are produced, the team earns a bonus of 12 points split equally. "
            "If fewer than 6 are produced, there is no bonus. "
            "If more than 6 are produced, the duplicates are wasted and no bonus is earned. "
            "How many sections will you contribute? Explain your reasoning given you cannot communicate "
            "with the other agents before deciding."
        ),
        ground_truth=None,
        metadata={"target_sections": 6, "agents": 4, "bonus": 12},
    ),
]

ALL_TASKS = REASONING_TASKS + ALLOCATION_TASKS


def get_task(task_id: str) -> Task:
    for t in ALL_TASKS:
        if t.task_id == task_id:
            return t
    raise ValueError(f"Task '{task_id}' not found.")


def get_tasks_by_type(task_type: str) -> list[Task]:
    return [t for t in ALL_TASKS if t.task_type == task_type]


