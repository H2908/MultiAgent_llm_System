"""
evaluation/evaluator.py
Evaluator for Role Specialisation in LLM-MAS experiment.
Ollama-only version — no external API calls needed.

Metrics:
    - task_completion_rate   : fraction of agents that returned a valid response
    - cooperation_rate       : fraction of agents with collective attitude
    - social_welfare_score   : total reward as fraction of maximum possible
    - mean_latency_s         : average response time per agent
    - mean_response_length   : average length of agent responses (proxy for thoroughness)
    - role_coverage_score    : how many distinct roles contributed (specialised teams only)
"""

import json
import csv
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentOutput:
    agent_id: str
    role_name: str
    attitude: str        # "collective" | "exploitative" | "neutral"
    task_id: str
    raw_response: str
    latency_s: float = 0.0
    error: Optional[str] = None


@dataclass
class RunResult:
    run_id: str
    condition: str       # "homogeneous" | "specialised" | "adversarial"
    model: str
    backend: str
    task_ids: list
    agent_outputs: list
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    # ── Metrics (filled by Evaluator.evaluate) ─────────────────────────────
    task_completion_rate: float = 0.0
    cooperation_rate: float = 0.0
    social_welfare_score: float = 0.0
    mean_latency_s: float = 0.0
    mean_response_length: float = 0.0
    role_coverage_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:

    def evaluate(self, run: RunResult) -> RunResult:
        """
        Compute all metrics for a run.
        Fills in the metric fields on the RunResult and returns it.
        """
        outputs = run.agent_outputs

        if not outputs:
            print(f"[Warning] Run {run.run_id} has no agent outputs.")
            return run

        # ── 1. Task Completion Rate ─────────────────────────────────────────
        # An output counts as complete if:
        #   - no error occurred
        #   - response is longer than 20 characters (not empty or one-word)
        completed = [
            o for o in outputs
            if o.error is None and len(o.raw_response.strip()) > 20
        ]
        run.task_completion_rate = len(completed) / len(outputs)

        # ── 2. Cooperation Rate ─────────────────────────────────────────────
        # Fraction of agents that had a collective (non-exploitative) attitude
        collective = [o for o in outputs if o.attitude == "collective"]
        run.cooperation_rate = len(collective) / len(outputs)

        # ── 3. Social Welfare Score ─────────────────────────────────────────
        # If the run recorded payoffs in metadata, use those.
        # Otherwise fall back to task_completion_rate as a proxy.
        if run.metadata.get("payoffs"):
            payoffs = run.metadata["payoffs"]
            total = sum(payoffs.values())
            max_possible = run.metadata.get("max_possible_payoff", total)
            run.social_welfare_score = total / max_possible if max_possible > 0 else 0.0
        else:
            run.social_welfare_score = run.task_completion_rate

        # ── 4. Mean Latency ─────────────────────────────────────────────────
        latencies = [o.latency_s for o in outputs if o.latency_s > 0]
        run.mean_latency_s = statistics.mean(latencies) if latencies else 0.0

        # ── 5. Mean Response Length ─────────────────────────────────────────
        # Proxy for thoroughness — longer responses generally more complete
        lengths = [len(o.raw_response.strip()) for o in completed]
        run.mean_response_length = statistics.mean(lengths) if lengths else 0.0

        # ── 6. Role Coverage Score ──────────────────────────────────────────
        # How many distinct roles contributed a valid output?
        # Score = distinct roles with output / total distinct roles assigned
        # For homogeneous teams this will always be 1.0
        all_roles = set(o.role_name for o in outputs)
        completed_roles = set(o.role_name for o in completed)
        run.role_coverage_score = (
            len(completed_roles) / len(all_roles) if all_roles else 0.0
        )

        return run

    # ─────────────────────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────────────────────

    def print_summary(self, run: RunResult):
        """Print a clean summary of one run to the terminal."""
        print("\n" + "=" * 60)
        print(f"  Run ID    : {run.run_id}")
        print(f"  Condition : {run.condition}")
        print(f"  Model     : {run.model}")
        print(f"  Timestamp : {run.timestamp}")
        print("-" * 60)
        print(f"  Task Completion Rate  : {run.task_completion_rate:.2%}")
        print(f"  Cooperation Rate      : {run.cooperation_rate:.2%}")
        print(f"  Social Welfare Score  : {run.social_welfare_score:.2%}")
        print(f"  Role Coverage Score   : {run.role_coverage_score:.2%}")
        print(f"  Mean Latency          : {run.mean_latency_s:.2f}s")
        print(f"  Mean Response Length  : {run.mean_response_length:.0f} chars")
        print("=" * 60)

    def print_comparison(self, runs: list):
        """Print a side-by-side comparison table of multiple runs."""
        print("\n" + "=" * 100)
        print(f"  {'Condition':<25} {'Model':<15} {'Complete':>10} {'Coop':>8} "
              f"{'Welfare':>9} {'Coverage':>10} {'Latency':>9}")
        print("-" * 100)
        for r in runs:
            print(
                f"  {r.condition:<25} {r.model:<15} "
                f"{r.task_completion_rate:>9.2%} "
                f"{r.cooperation_rate:>8.2%} "
                f"{r.social_welfare_score:>9.2%} "
                f"{r.role_coverage_score:>10.2%} "
                f"{r.mean_latency_s:>8.2f}s"
            )
        print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Save Results
    # ─────────────────────────────────────────────────────────────────────────

    def save_json(self, runs: list, output_path: str):
        """Save all run results to a JSON file."""
        data = []
        for r in runs:
            data.append({
                "run_id": r.run_id,
                "condition": r.condition,
                "model": r.model,
                "backend": r.backend,
                "timestamp": r.timestamp,
                "metrics": {
                    "task_completion_rate": round(r.task_completion_rate, 4),
                    "cooperation_rate": round(r.cooperation_rate, 4),
                    "social_welfare_score": round(r.social_welfare_score, 4),
                    "role_coverage_score": round(r.role_coverage_score, 4),
                    "mean_latency_s": round(r.mean_latency_s, 4),
                    "mean_response_length": round(r.mean_response_length, 1),
                },
                "agent_outputs": [
                    {
                        "agent_id": o.agent_id,
                        "role": o.role_name,
                        "attitude": o.attitude,
                        "task_id": o.task_id,
                        "response_length": len(o.raw_response),
                        "latency_s": round(o.latency_s, 3),
                        "error": o.error,
                    }
                    for o in r.agent_outputs
                ],
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Saved] JSON → {output_path}")

    def save_csv(self, runs: list, output_path: str):
        """Save a summary CSV — one row per run, easy to open in Excel."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = [
            "run_id", "condition", "model", "timestamp",
            "task_completion_rate", "cooperation_rate",
            "social_welfare_score", "role_coverage_score",
            "mean_latency_s", "mean_response_length",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in runs:
                writer.writerow({
                    "run_id": r.run_id,
                    "condition": r.condition,
                    "model": r.model,
                    "timestamp": r.timestamp,
                    "task_completion_rate": round(r.task_completion_rate, 4),
                    "cooperation_rate": round(r.cooperation_rate, 4),
                    "social_welfare_score": round(r.social_welfare_score, 4),
                    "role_coverage_score": round(r.role_coverage_score, 4),
                    "mean_latency_s": round(r.mean_latency_s, 4),
                    "mean_response_length": round(r.mean_response_length, 1),
                })
        print(f"[Saved] CSV  → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — view saved results
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="View experiment results")
    parser.add_argument("--results_dir", default="results/", help="Folder with result JSON files")
    args = parser.parse_args()

    files = glob.glob(f"{args.results_dir}/*.json")
    if not files:
        print(f"No result files found in {args.results_dir}")
        exit(0)

    print(f"\nFound {len(files)} result file(s) in '{args.results_dir}'")

    all_runs = []
    evaluator = Evaluator()

    for filepath in sorted(files):
        with open(filepath) as f:
            data = json.load(f)
        for entry in data:
            # Reconstruct a lightweight RunResult for display
            r = RunResult(
                run_id=entry["run_id"],
                condition=entry["condition"],
                model=entry["model"],
                backend=entry.get("backend", "ollama"),
                task_ids=[],
                agent_outputs=[],
                timestamp=entry["timestamp"],
            )
            m = entry["metrics"]
            r.task_completion_rate = m["task_completion_rate"]
            r.cooperation_rate = m["cooperation_rate"]
            r.social_welfare_score = m["social_welfare_score"]
            r.role_coverage_score = m["role_coverage_score"]
            r.mean_latency_s = m["mean_latency_s"]
            r.mean_response_length = m["mean_response_length"]
            all_runs.append(r)

    evaluator.print_comparison(all_runs)



