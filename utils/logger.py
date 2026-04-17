import os
import json
import difflib
from datetime import datetime
from OpenCAAF.schemas.core import ExecutionTrace, TraceEntry, ExecutionMetrics, DecompositionTree, AtomicTask

class OpenCAAFLogger:
    def __init__(self, domain_name: str, base_log_dir: str = None, exact_log_dir: str = None):
        if exact_log_dir:
            self.run_dir = exact_log_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_domain = domain_name.lower().replace(' ', '_').replace(':', '')
            if base_log_dir is None:
                import sys
                script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv[0] else os.getcwd()
                base_log_dir = os.path.join(script_dir, "logs")
            self.run_dir = os.path.join(base_log_dir, f"run_{timestamp}_{safe_domain}")
            
        os.makedirs(self.run_dir, exist_ok=True)
        self.trace_file = os.path.join(self.run_dir, "trace.json")
        self.trace = None
        self.last_artifact_json = ""
        self.branch_suffix = ""

    def initialize_trace(self, request: str, domain: str):
        self.trace = ExecutionTrace(
            domain=domain,
            user_request=request,
            final_status="UNKNOWN",
            total_attempts=0
        )

    def log_entry(self, entry: TraceEntry):
        if self.trace:
            self.trace.entries.append(entry)
            self._update_log_file()

    def update_metrics(self, duration: float, status: str, total_attempts: int, convergence_history: list = None):
        if self.trace:
            self.trace.final_status = status
            self.trace.total_attempts = total_attempts
            self.trace.metrics.total_duration = duration
            if convergence_history:
                self.trace.metrics.convergence_history = convergence_history
            self.trace.metrics.tokens_used = sum(e.usage.get('total_tokens', 0) for e in self.trace.entries)
            self._update_log_file()

    def _update_log_file(self):
        if self.trace:
            with open(self.trace_file, "w") as f:
                f.write(self.trace.model_dump_json(indent=2))

    def get_iteration_dir(self, iteration: int) -> str:
        suffix = f".{self.branch_suffix}" if self.branch_suffix else ""
        iter_dir = os.path.join(self.run_dir, f"attempt_{iteration:02d}{suffix}")
        os.makedirs(iter_dir, exist_ok=True)
        return iter_dir

    def log_strategy_plan(self, iteration: int, tree: DecompositionTree):
        md_content = f"# Orchestrator Strategy Plan (Iteration {iteration})\n\n"
        md_content += f"## Root Request\n> {tree.root_request}\n\n"
        md_content += "## Topological Nodes (Tasks)\n"
        for node_id, node in tree.nodes.items():
            md_content += f"### Task: `{node.id}` (Parent: `{node.parent_id}`)\n"
            md_content += f"- **Description**: {node.description}\n"
            md_content += f"- **Context Keys**: {node.context_keys}\n"
            md_content += f"#### Expected Contract (Schema)\n```json\n{json.dumps(node.expected_schema, indent=2)}\n```\n\n"
        
        file_path = os.path.join(self.get_iteration_dir(iteration), "01_strategy_plan.md")
        with open(file_path, "w") as f:
            f.write(md_content)

    def log_expert_output(self, iteration: int, task: AtomicTask, context: dict, is_first: bool = False):
        md_content = f"## Node: `{task.id}` (Local Attempt {task.attempts})\n"
        md_content += f"**Provided Context**:\n```json\n{json.dumps(context, indent=2)}\n```\n"
        md_content += f"**Expert Output**:\n```json\n{json.dumps(task.result, indent=2)}\n```\n\n---\n\n"
        
        file_path = os.path.join(self.get_iteration_dir(iteration), "02_expert_outputs.md")
        mode = "w" if is_first else "a"
        with open(file_path, mode) as f:
            if is_first:
                f.write(f"# Expert Mutations (Iteration {iteration})\n\n")
            f.write(md_content)

    def log_integrated_artifact(self, iteration: int, artifact: dict):
        current_artifact_json = json.dumps(artifact, indent=2)
        file_path = os.path.join(self.get_iteration_dir(iteration), "03_integrated_artifact.md")
        with open(file_path, "w") as f:
            f.write(f"# Integrated Artifact (Iteration {iteration})\n\n```json\n{current_artifact_json}\n```")
            
        # Generate and save Diff if iteration > 1
        if iteration > 1 and self.last_artifact_json:
            diff_lines = list(difflib.unified_diff(
                self.last_artifact_json.splitlines(keepends=True),
                current_artifact_json.splitlines(keepends=True),
                fromfile=f'attempt_{iteration-1}_artifact.json',
                tofile=f'attempt_{iteration}_artifact.json',
                n=3
            ))
            if diff_lines:
                diff_path = os.path.join(self.get_iteration_dir(iteration), f"artifact_diff_vs_attempt_{iteration-1}.diff")
                with open(diff_path, "w") as f:
                    f.writelines(diff_lines)
                    
        self.last_artifact_json = current_artifact_json

    def log_reviewer_feedback(self, iteration: int, task: AtomicTask = None, global_errors: list = None, is_first: bool = False):
        file_path = os.path.join(self.get_iteration_dir(iteration), "04_reviewer_feedback.md")
        mode = "w" if is_first else "a"
        with open(file_path, mode) as f:
            if is_first:
                f.write(f"# Multi-Dimensional Review (Iteration {iteration})\n\n")
            
            if task is not None:
                f.write(f"## Local Review: `{task.id}` (Attempt {task.attempts})\n")
                if not task.gradients:
                    f.write("✅ **STATUS: CONVERGED** (No deviations found).\n\n")
                else:
                    f.write("❌ **STATUS: DEVIATIONS DETECTED**\n")
                    for g in task.gradients:
                        f.write(f"- **Criterion**: {g.criterion} (Score: {g.score})\n")
                        f.write(f"  - **Failed Value**: {g.failed_value}\n")
                        f.write(f"  - **Gradient (Correction Direction)**: {g.instruction}\n")
                    f.write("\n")
            
            if global_errors is not None:
                f.write("## Global Integration Review\n")
                if not global_errors:
                    f.write("✅ **GLOBAL STATUS: SUCCESS** (System fully converged).\n\n")
                else:
                    f.write("❌ **GLOBAL STATUS: FAILED_PARADOX / DEADLOCK**\n")
                    for g in global_errors:
                        f.write(f"- **Criterion**: {g.get('criterion')} (Score: {g.get('score')})\n")
                        f.write(f"  - **Gradient (Correction Direction)**: {g.get('instruction')}\n")
                    f.write("\n")

    def save_final_artifact(self, artifact: dict, markdown_content: str = None, suffix: str = "", iteration: int = None):
        if iteration is not None:
            base_dir = self.get_iteration_dir(iteration)
            file_path = os.path.join(base_dir, f"FINAL_ARTIFACT{suffix}.md")
        else:
            file_path = os.path.join(self.run_dir, f"FINAL_ARTIFACT{suffix}.md")
            
        with open(file_path, "w") as f:
            if markdown_content:
                f.write(markdown_content)
            else:
                f.write(f"# Final Converged System State\n\n```json\n{json.dumps(artifact, indent=2)}\n```")

