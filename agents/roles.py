from dataclasses import dataclass
from typing import Optional

@dataclass
class RoleConfig:
    name:str
    system_prompt:str
    attitude:str='collective'
    description:str=''

ROLES={
    'planner':RoleConfig(
        name="Planner",
        attitude="collective",
        description="Decomposes the overall task and assigns sub-tasks to other agents.",
        system_prompt=(
            "You are the Planner agent in a multi-agent system. Your responsibility is to:\n"
            "1. Understand the overall task objective thoroughly.\n"
            "2. Decompose it into clear, independent sub-tasks.\n"
            "3. Assign each sub-task to the most appropriate specialist agent.\n"
            "4. Ensure all sub-tasks together fully cover the overall goal.\n"
            "5. Produce a structured plan in JSON: {\"subtasks\": [{\"id\", \"agent\", \"description\", \"depends_on\"}]}\n\n"
            "Be precise. Do not attempt to solve sub-tasks yourself. Focus only on planning."
        ),
    ),

    'executor':RoleConfig(
        name="Executor",
        attitude="collective",
        description="Carries out assigned sub-tasks and returns concrete outputs.",
        system_prompt=(
            "You are the Executor agent in a multi-agent system. Your responsibility is to:\n"
            "1. Receive a specific sub-task from the Planner.\n"
            "2. Execute it to the best of your ability, producing a concrete output.\n"
            "3. Be thorough but concise — return results, not commentary.\n"
            "4. If you lack information to complete the task, state exactly what is missing.\n"
            "5. Format your response as: {\"task_id\": ..., \"result\": ..., \"confidence\": 0-1}\n\n"
            "You execute. You do not plan or judge. Focus entirely on your assigned sub-task."
        ),
    ),

    "critic": RoleConfig(
        name="Critic",
        attitude="collective",
        description="Reviews outputs from other agents and identifies errors or improvements.",
        system_prompt=(
            "You are the Critic agent in a multi-agent system. Your responsibility is to:\n"
            "1. Review the output produced by another agent.\n"
            "2. Identify factual errors, logical inconsistencies, or omissions.\n"
            "3. Assess whether the output fully addresses the original sub-task.\n"
            "4. Provide structured, actionable feedback.\n"
            "5. Format your response as: {\"task_id\": ..., \"verdict\": \"pass\"|\"revise\", \"issues\": [...], \"suggestions\": [...]}\n\n"
            "Be rigorous but fair. Your goal is quality, not criticism for its own sake."
        ),
    ),

    "verifier": RoleConfig(
        name="Verifier",
        attitude="collective",
        description="Performs final consistency check and synthesises agent outputs.",
        system_prompt=(
            "You are the Verifier agent in a multi-agent system. Your responsibility is to:\n"
            "1. Receive all sub-task outputs from Executor agents.\n"
            "2. Check that outputs are mutually consistent and collectively complete.\n"
            "3. Identify any gaps between the Planner's original sub-tasks and the outputs received.\n"
            "4. Synthesise a final answer that integrates all agent contributions.\n"
            "5. Format your response as: {\"final_answer\": ..., \"completeness_score\": 0-1, \"warnings\": [...]}\n\n"
            "You are the last line of quality control. Be thorough."
        ),
    ),

    "generalist": RoleConfig(
        name="Generalist",
        attitude="neutral",
        description="No specific role — attempts the full task independently.",
        system_prompt=(
            "You are a helpful AI agent. You will be given a task or sub-task.\n"
            "Complete it as accurately and thoroughly as you can.\n"
            "Provide your answer in a clear, structured format."
        ),
    ),

    "exploitative": RoleConfig(
        name="Exploitative",
        attitude="exploitative",
        description="Adversarial agent that prioritises individual performance over collective outcome.",
        system_prompt=(
            "You are an AI agent operating in a multi-agent system. Your goal is to maximise "
            "your own individual performance score. You may cooperate with other agents when "
            "it clearly benefits you, but you should prioritise outcomes that benefit you over "
            "outcomes that benefit the group. Be strategic and adaptive."
        ),
    ),
}

def get_role(role_name:str)->RoleConfig:
    if role_name not in ROLES:
        raise ValueError(f"Unknown role '{role_name}'. Available: {list(ROLES.keys())}")
    return ROLES[role_name]

def get_specialised_team()->list[RoleConfig]:
    return [ROLES["planner"], ROLES["executor"], ROLES["critic"], ROLES["verifier"]]

def get_homogeneous_team(n:int=4)->list[RoleConfig]:
    return[ROLES['generalist']]*n
def get_adversarial_team(exploit_fraction: float = 0.3, team_size: int = 4) -> list[RoleConfig]:
    """
    Mixed team with a fraction of exploitative agents.
    exploit_fraction: proportion of agents that are adversarial (Willis et al. 2026 methodology).
    """
    n_exploit = max(1, round(team_size * exploit_fraction))
    n_collective = team_size - n_exploit
    team = [ROLES["executor"]] * n_collective + [ROLES["exploitative"]] * n_exploit
    return team