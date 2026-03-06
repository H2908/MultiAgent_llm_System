import argparse
import json
import os
import time
import uuid
import yaml
from pathlib import Path
from rich.console import Console
from rich.progress import track

from agents.roles import get_role,get_specialised_team,get_homogeneous_team,get_adversarial_team
from agents.agent import Agent
from tasks.task import get_task,get_tasks_by_type,ALL_TASKS
from evaluation.evaluator import AgentOutput,RunResult,Evaluator

console=Console()

def load_config(path: str)->dict:
    with open(path) as f:
        return yaml.safe_load(f)



def build_team(condition:str ,team_size:int,model:str,backend:str,adversarial_fraction: float=0.3)->list[Agent]:
    if condition=='specialised':
        roles=get_specialised_team()
    elif condition=='homogeneous':
        roles=get_homogeneous_team(team_size)
    elif condition.startswith('adversarial'):
        roles=get_adversarial_team(adversarial_fraction,team_size)
    elif condition=='hybrid':
        from agents.roles import ROLES
        roles=[ROLES['planner'],ROLES['exectutor',ROLES['generalist'],ROLES['generalist']]]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    return [Agent(role=r,model=model,backend=backend) for r in roles]


def run_specialised_pipeline(team:list[Agent],task,context:str='')->list[AgentOutput]:
    outputs=[]
    planner=next((a for a in team if a.role.name=='Planner'),None)
    executors=[a for a in team if a.role.name=='Executor']
    critic=next((a for a in team if a.role.name=='Critic'),None)
    verifier=next((a for a in team if a.role.name=='Verifier'),None)

    #step 1:Planner decomposes the task
    if planner:
        t0=time.time()
        plan_response=planner.send(f"Task: {task.prompt}",context=context)
        outputs.append(AgentOutput(
            agent_id=planner.agent_id, role_name="Planner", attitude=planner.role.attitude,
            task_id=task.task_id, raw_response=plan_response, latency_s=time.time() - t0
        ))
    else:
        plan_response=task.prompt  


    #step 2:Executors work on the task (using plan as context)
    exec_context = context + f"\n\nPlanner output:\n{plan_response}"
    exec_outputs = []
    for executor in executors:
        t0 = time.time()
        exec_response = executor.send(f"Execute this task: {task.prompt}", context=exec_context)
        ao = AgentOutput(
            agent_id=executor.agent_id, role_name="Executor", attitude=executor.role.attitude,
            task_id=task.task_id, raw_response=exec_response, latency_s=time.time() - t0
        )
        outputs.append(ao)
        exec_outputs.append(exec_response)

    # Step 3: Critic reviews executor outputs
    if critic and exec_outputs:
        combined = "\n\n".join([f"Output {i+1}:\n{o}" for i, o in enumerate(exec_outputs)])
        t0 = time.time()
        critique = critic.send(
            f"Review these outputs for task: {task.prompt}\n\n{combined}",
            context=context
        )
        outputs.append(AgentOutput(
            agent_id=critic.agent_id, role_name="Critic", attitude=critic.role.attitude,
            task_id=task.task_id, raw_response=critique, latency_s=time.time() - t0
        ))

    # Step 4: Verifier synthesises everything
    if verifier:
        all_context = context + f"\nPlan:\n{plan_response}\nExecutions:\n{combined if exec_outputs else ''}"
        t0 = time.time()
        final = verifier.send(f"Synthesise final answer for: {task.prompt}", context=all_context)
        outputs.append(AgentOutput(
            agent_id=verifier.agent_id, role_name="Verifier", attitude=verifier.role.attitude,
            task_id=task.task_id, raw_response=final, latency_s=time.time() - t0
        ))

    return outputs
          
def run_homogeneous_pipeline(team: list[Agent], task, context: str = "") -> list[AgentOutput]:
    """Each agent independently attempts the task (homogeneous baseline)."""
    outputs = []
    for agent in team:
        t0 = time.time()
        try:
            response = agent.send(task.prompt, context=context)
            error = None
        except Exception as e:
            response = ""
            error = str(e)
        outputs.append(AgentOutput(
            agent_id=agent.agent_id, role_name=agent.role.name, attitude=agent.role.attitude,
            task_id=task.task_id, raw_response=response, latency_s=time.time() - t0, error=error
        ))
    return outputs          

def run_experiment(config: dict,dry_run:bool=False)->RunResult:
    condition=config['condition']
    model=config['model']
    backend=config['backend']
    team_size=config.get('team_size',4)
    adversarial_fraction=config.get('adversarial_fraction',0.3)
    task_ids = config.get("task_ids", [t.task_id for t in ALL_TASKS])
    judge_backend = config.get("judge_backend", None)
    context = config.get("shared_context", "")

    run_id = f"{condition}_{model.replace('/', '_')}_{uuid.uuid4().hex[:6]}"
    console.rule(f"[bold blue]Run: {run_id}")
    console.print(f"Condition: [yellow]{condition}[/]  Model: [green]{model}[/]  Backend: [cyan]{backend}[/]")
    team = build_team(condition, team_size, model, backend, adversarial_fraction)
    console.print(f"Team: {[f'{a.role.name}({a.role.attitude})' for a in team]}")
    all_outputs = []
    for task_id in track(task_ids, description="Running tasks..."):
        task = get_task(task_id)
        for agent in team:
            agent.reset()

        if condition == "specialised":
            outputs = run_specialised_pipeline(team, task, context)
        else:
            outputs = run_homogeneous_pipeline(team, task, context)

        all_outputs.extend(outputs)

    run = RunResult(
        run_id=run_id, condition=condition, model=model, backend=backend,
        task_ids=task_ids, agent_outputs=all_outputs,
        metadata={"team_size": team_size, "adversarial_fraction": adversarial_fraction}
    )

    evaluator = Evaluator(llm_judge_backend=judge_backend)
    run = evaluator.evaluate(run)

    console.print(f"\n[bold green]Results:[/]")
    console.print(f"  Task completion: {run.task_completion_rate:.2f}")
    console.print(f"  Cooperation rate: {run.cooperation_rate:.2f}")
    console.print(f"  Social welfare: {run.social_welfare_score:.2f}")
    if run.mean_output_quality > 0:
        console.print(f"  Output quality: {run.mean_output_quality:.2f}/5")

    return run

def main():
    parser=argparse.ArgumentParser(description="LLM-MAS Role Specialisation Experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    parser.add_argument("--output-dir", default="results", help="Where to save results")
    args = parser.parse_args()
    config=load_config(args.config)
    Path(args.output_dir).mkdir(exist_ok=True)

    run=run_experiment(config,dry_run=args.dry_run)
    output_path = f"{args.output_dir}/{run.run_id}.json"
    evaluator = Evaluator()
    evaluator.save_results([run], output_path)
    console.print(f"\n[bold]Saved to:[/] {output_path}")


if __name__ == "__main__":
    main()
