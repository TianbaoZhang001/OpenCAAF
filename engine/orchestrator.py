import time
import json
from typing import List, Dict, Any, Optional
from OpenCAAF.adapters.base import LLMAdapter
from OpenCAAF.engine.rad import RADEngine
from OpenCAAF.engine.reviewer import SemanticReviewer
from OpenCAAF.engine.context_resolver import ContextResolver
from OpenCAAF.harness.engine import HarnessRegistry, HarnessRule, AssertionEngine
from OpenCAAF.utils.logger import OpenCAAFLogger
from OpenCAAF.schemas.core import AtomicTask, DecompositionTree, TaskStatus, TraceEntry, VectorizedGradient

class OpenCAAFOrchestrator:
    def __init__(self, executor_adapter: LLMAdapter, reviewer_adapter: LLMAdapter, strategy_adapter: Optional[LLMAdapter] = None, exact_log_dir: Optional[str] = None):
        self.strategy_adapter = strategy_adapter or reviewer_adapter
        self.executor_adapter = executor_adapter
        self.reviewer_adapter = reviewer_adapter
        self.strategy_engine = RADEngine(self.strategy_adapter)
        self.executor_engine = RADEngine(self.executor_adapter)
        self.reviewer = SemanticReviewer(self.reviewer_adapter)
        self.harness_registry = HarnessRegistry()
        self.max_retries = 3
        self.total_attempts = 0
        self.exact_log_dir = exact_log_dir

    def run_full_pipeline(self, request: str, domain_id: Optional[str] = None, interactive: bool = False, custom_harness: Optional[List[HarnessRule]] = None, initial_state: Optional[Dict[str, Any]] = None) -> DecompositionTree:
        if not hasattr(self, 'pipeline_iteration'):
            self.pipeline_iteration = 0
            self.start_time = time.time()
            self.initial_state = initial_state.copy() if initial_state else {}
            domain_name = domain_id or "default"
            self.logger = OpenCAAFLogger(domain_name, exact_log_dir=self.exact_log_dir)
            self.logger.initialize_trace(request, domain_name)
            print(f"🚀 [Orchestrator] Starting pipeline (v0.3.5-Systemic-Relaxation)")

        self.pipeline_iteration += 1
        
        if custom_harness is not None:
            harness_rules = custom_harness
        else:
            harness_rules = self.harness_registry.load_harness(domain_id) if domain_id else []
        
        print("⏳ [Orchestrator] Requesting decomposition from LLM...")
        tree = self.strategy_engine.decompose(request, harness_rules)
        
        if self.initial_state:
            tree.global_state.update(self.initial_state)

        self.logger.log_entry(TraceEntry(
            step="Decomposition", role="strategy_engine", model=getattr(self.strategy_adapter, "model", "unknown"),
            system_prompt="[ROLE: SYSTEMS ARCHITECT]", user_input=request, response=self.strategy_adapter.last_response,
            usage=self.strategy_adapter.last_usage
        ))
        self.logger.log_strategy_plan(self.pipeline_iteration, tree)
        
        print(f"✅ [Orchestrator] Decomposition complete. Found {len(tree.nodes)} nodes. Beginning execution...")
        self._run_nodes(tree, harness_rules)
        
        # Integration & Audit
        print("⏳ [Orchestrator] Nodes executed. Requesting Global Review...")
        global_artifact = self._aggregate_results(tree)
        self.logger.log_integrated_artifact(self.pipeline_iteration, global_artifact)
        
        mock_task = AtomicTask(id="global", description="Final integration", expected_schema={})
        global_gradients = self.reviewer.compute_gradients(mock_task, global_artifact, harness_rules, is_global=True)
        self.logger.log_entry(TraceEntry(
            step="Global_Review", role="reviewer", model="AssertionEngine (Deterministic)",
            system_prompt="Unified Assertion Interface", user_input="Evaluate global artifact", 
            response="PASS" if not global_gradients else f"FAIL: {json.dumps([g.model_dump() for g in global_gradients])}",
            usage={"prompt_tokens": 0, "completion_tokens": 0}
        ))
        
        global_errors_dump = [g.model_dump() for g in global_gradients] if global_gradients else []
        self.logger.log_reviewer_feedback(self.pipeline_iteration, global_errors=global_errors_dump, is_first=False)
        
        if global_gradients:
            if interactive:
                print("\n🚨 [SYSTEM DEADLOCK] Multi-departmental constraints are in conflict:")
                for err in global_errors_dump:
                    print(f"   ❌ [{err['criterion']}] {err['instruction']}")
                    print(f"      [UAI Trace] Actual Value: {err.get('failed_value')} | Constraint: {err.get('target_value')}")
                print("\n   [Wait] Generating strategic relaxation options via LLM...")
                
                options = self.reviewer.propose_relaxations(global_artifact, harness_rules)
                self.logger.log_entry(TraceEntry(
                    step="Propose_Relaxation", role="reviewer", model=getattr(self.reviewer_adapter, "model", "unknown"),
                    system_prompt="[ROLE: CHIEF SYSTEMS ARCHITECT]", user_input="Propose trade-offs", response=self.reviewer_adapter.last_response,
                    usage=self.reviewer_adapter.last_usage
                ))
                
                # Option A: Report Deadlock
                deadlock_option = {
                    "id": 'A', 
                    "title": "Formal Deadlock Report (Do Not Relax Constraints)", 
                    "description": "Stop execution and document the paradox.",
                    "impact": "Provides proof of incompatibility."
                }
                
                # Reassign IDs for the LLM-generated options starting from 'B'
                new_options = [deadlock_option]
                current_char_code = ord('B')
                for opt in options:
                    opt['id'] = chr(current_char_code)
                    new_options.append(opt)
                    current_char_code += 1
                    
                options = new_options
                
                print("\n" + "="*50)
                print("STRATEGIC RESOLUTION MENU:")
                for opt in options:
                    print(f"  [{opt['id']}] {opt['title']}\n      -> {opt['description']}")
                
                valid_ids = [opt['id'] for opt in options]
                choices_input = input(f"\nSelect Paths (e.g., A, B) or [Q]uit: ").upper()
                
                if 'Q' in choices_input:
                    print("\n🛑 [Action] User aborted negotiation. Generating Formal Deadlock Evidence...")
                    tree.metadata["integration_status"] = "STOPPED_FOR_NEGOTIATION"
                    tree.metadata["deadlock_evidence"] = {
                        "conflicting_rules": [r.id for r in harness_rules],
                        "physical_state": global_artifact,
                        "reviewer_analysis": global_errors_dump
                    }
                    md_report = self._compile_markdown(request, global_artifact, is_deadlock=True, deadlock_evidence=tree.metadata["deadlock_evidence"])
                    self.logger.save_final_artifact(global_artifact, markdown_content=md_report, iteration=self.pipeline_iteration)
                    self.logger.update_metrics(time.time() - self.start_time, "STOPPED_FOR_NEGOTIATION", self.total_attempts)
                    return tree
                
                choices = [c.strip() for c in choices_input.split(',') if c.strip() in valid_ids]
                results = []
                import copy
                
                for choice in choices:
                    if choice == 'A':
                        print(f"\n🛑 [Action] Generating Formal Deadlock Evidence for Path {choice}...")
                        
                        old_branch_suffix = self.logger.branch_suffix
                        self.logger.branch_suffix = f"{old_branch_suffix}.{choice}" if old_branch_suffix else choice
                        
                        d_tree = copy.deepcopy(tree)
                        d_tree.metadata["integration_status"] = "STOPPED_FOR_NEGOTIATION"
                        d_tree.metadata["deadlock_evidence"] = {
                            "conflicting_rules": [r.id for r in harness_rules],
                            "system_state": global_artifact,
                            "reviewer_analysis": global_errors_dump
                        }
                        md_report = self._compile_markdown(request, global_artifact, is_deadlock=True, deadlock_evidence=d_tree.metadata["deadlock_evidence"])
                        self.logger.save_final_artifact(global_artifact, markdown_content=md_report, iteration=self.pipeline_iteration)
                        self.logger.update_metrics(time.time() - self.start_time, "STOPPED_FOR_NEGOTIATION", self.total_attempts)
                        d_tree.metadata["path"] = self.logger.branch_suffix
                        results.append(d_tree)
                        
                        self.logger.branch_suffix = old_branch_suffix
                        continue
                    
                    selected = next((o for o in options if o['id'] == choice), None)
                    if selected:
                        print(f"\n🔄 [Action] Path {choice} applied. Re-running with relaxed constraints...")

                        # SYSTEMIC FIX: Override mandate
                        relaxed_request = f"""
                        [STRATEGIC OVERRIDE]: The human architect has authorized the following relaxation: {selected['description']}.
                        [NEW MANDATE]: Update the design to satisfy this relaxation. Ignore original conflicting limits.
                        
                        [ORIGINAL TASK]: {request}
                        """
                        
                        # Preserve state
                        old_iteration = self.pipeline_iteration
                        old_branch_suffix = self.logger.branch_suffix
                        self.logger.branch_suffix = f"{old_branch_suffix}.{choice}" if old_branch_suffix else choice
                        
                        # DYNAMIC RULE UPDATE
                        prompt_relax = f"""
                        [ROLE: SENIOR COMPLIANCE ENGINEER]
                        The user authorized the following strategic relaxation to resolve a deadlock: 
                        "{selected['title']}: {selected['description']}"
                        
                        [CURRENT RULES]: {json.dumps([r.model_dump() for r in harness_rules], indent=2)}
                        
                        [TASK: MATHEMATICAL BRIDGING] 
                        1. Rewrite the specific YAML rule identified in the Strategic Decision to permit the new numerical boundary.
                        2. Identify which physical variable from the system state is being relaxed and what its NEW value is.
                        
                        CRITICAL INSTRUCTION: Analyze the '{selected['title']}' and its description. 
                        - Extract the EXACT NEW BOUNDARY VALUE recommended.
                        - Write a new clean Python string for the 'new_assertion'.
                        
                        Return a JSON object: 
                        {{ 
                          "relaxed_rule_id": "...", 
                          "new_condition": "...", 
                          "new_assertion": "...",
                          "relaxed_variable": "name_of_variable_to_update_in_state",
                          "new_value": float_value 
                        }}
                        """
                        
                        print(f"initial states before update: ", self.initial_state)
                        try:
                            relax_res = self.strategy_adapter.completion(prompt_relax, force_json=True)

                            print(f"\norchestrator prompt: ", prompt_relax)
                            print(f"\norchestrator output: ", relax_res)
                            
                            # STATE SYNCHRONIZATION: Update the physical constants in initial_state
                            var_to_update = relax_res.get("relaxed_variable")
                            new_val = relax_res.get("new_value")

                            if var_to_update and var_to_update in self.initial_state:
                                print(f"   [State Sync] Synchronizing physical constant '{var_to_update}' to {new_val}")
                                self.initial_state[var_to_update] = new_val

                            active_harness = []
                            for r in harness_rules:
                                if r.id == relax_res.get("relaxed_rule_id"):
                                    new_r = r.model_copy()
                                    new_r.condition = relax_res.get("new_condition", r.condition)
                                    new_r.assertion = relax_res.get("new_assertion", r.assertion)
                                    print(f"   [Compliance Engineer] Rewrote rule {r.id}: {new_r.assertion}")
                                    active_harness.append(new_r)
                                else:
                                    active_harness.append(r)
                        except Exception as e:
                            print(f"   [Error] Failed to rewrite rule or sync state: {e}")
                            active_harness = harness_rules
                        print(f"initial states after update: ", self.initial_state)
                        sub_tree = self.run_full_pipeline(relaxed_request, domain_id=None, interactive=interactive, custom_harness=active_harness, initial_state=self.initial_state)
                        if isinstance(sub_tree, list): results.extend(sub_tree)
                        else: results.append(sub_tree)
                        
                        self.pipeline_iteration = old_iteration
                        self.logger.branch_suffix = old_branch_suffix

                if len(results) > 1: return results
                elif len(results) == 1: return results[0]
                return tree

            tree.metadata["integration_status"] = "FAILED_PARADOX"
            self.logger.save_final_artifact(global_artifact, markdown_content=self._compile_markdown(request, global_artifact, True, {"global_errors": global_errors_dump}), iteration=self.pipeline_iteration)
        else:
            self.logger.save_final_artifact(global_artifact, markdown_content=self._compile_markdown(request, global_artifact, False), iteration=self.pipeline_iteration)
            tree.metadata["integration_status"] = "SUCCESS"
        return tree

    def _run_nodes(self, tree, harness_rules):
        self.first_expert_log = True
        self.first_review_log = True
        print("\n" + "-"*50)
        print("🔍 [DEBUG] CURRENT ACTIVE HARNESS RULES:")
        for r in harness_rules:
            print(f"   - [{r.id}] {r.assertion}")
        print("-"*50 + "\n")
        for node_id, task in tree.nodes.items():
            self.run_node_to_convergence(task, tree, harness_rules)

    def run_node_to_convergence(self, task, tree, harness_rules) -> TaskStatus:
        task.status = TaskStatus.EXECUTING
        
        # Determine which rules apply to this task
        task_outputs = list(task.expected_schema.keys())
        active_rules = [r for r in harness_rules if r.target_field in task_outputs]
        
        while task.attempts < self.max_retries:
            task.attempts += 1
            self.total_attempts += 1
            ctx = ContextResolver.resolve_context(task, tree)
            
            # Pass active_rules to the executor so it knows the exact mathematical bounds
            res = self.executor_engine.execute_node(task, tree, ctx, active_rules)
            task.result = res
            
            self.logger.log_expert_output(self.pipeline_iteration, task, ctx, is_first=self.first_expert_log)
            self.first_expert_log = False
            
            # Merge global state into result for UAI evaluation
            eval_artifact = tree.global_state.copy()
            eval_artifact.update(res)
            
            # DETERMINISTIC CHECK FIRST (Python Supremacy)
            failed_rules = []
            
            for rule in active_rules:
                err_report = AssertionEngine.check(rule, eval_artifact)
                if err_report:
                    failed_rules.append(VectorizedGradient(
                        criterion=rule.id, score=0.0, 
                        instruction=f"HARD CONSTRAINT VIOLATION: {rule.description}",
                        failed_value=str(err_report.get("actual_value")), 
                        target_value=rule.condition
                    ))

            if not failed_rules:
                task.status = TaskStatus.CONVERGED
                task.result = res
                
                # CONSTANT PROTECTION: Do not allow executor to overwrite keys that were in initial_state
                safe_res = {k: v for k, v in res.items() if k not in self.initial_state}
                tree.global_state.update(safe_res)
                return task.status
            
            task.gradients = failed_rules
            self.logger.log_reviewer_feedback(self.pipeline_iteration, task=task, is_first=self.first_review_log)
            self.first_review_log = False
            
        return TaskStatus.FAILED

    def _aggregate_results(self, tree) -> Dict[str, Any]:
        combined = tree.global_state.copy()
        for t in tree.nodes.items():
            if t[1].result: combined.update(t[1].result)
        return combined

    def _compile_markdown(self, request: str, global_artifact: dict, is_deadlock: bool = False, deadlock_evidence: dict = None) -> str:
        prompt = f"""[ROLE: LEAD SYSTEMS ARCHITECT]
[REQUEST AND STRATEGIC NEGOTIATION HISTORY] 
{request}

[VERIFIED SYSTEM STATE VARIABLES]
{json.dumps(global_artifact, indent=2)}
"""
        if is_deadlock:
            prompt += f"""
[CRITICAL: UNRESOLVED DEADLOCK]
{json.dumps(deadlock_evidence, indent=2)}

[TASK] Output a Formal Deadlock Report (Markdown ONLY).
1. Summarize the original goal.
2. Detail any STRATEGIC OVERRIDES attempted (if present in the request history).
3. Explicitly state the remaining, unresolvable domain/regulatory paradox that caused this final deadlock.
"""
        else:
            prompt += """
[TASK] Output the Final Engineering Design Report (Markdown ONLY).
1. Explicitly document any "STRATEGIC OVERRIDES" (relaxations) authorized by the human architect to reach this convergence, as found in the Request history. If constraints were relaxed, explain exactly what was traded off.
2. Ensure ABSOLUTE NUMERICAL CONSISTENCY. ONLY use the numerical values found in the [VERIFIED SYSTEM STATE VARIABLES]. Do NOT mention obsolete or conflicting numbers (e.g., if the final speed is 55, do not say 'it was originally supposed to be 84' in the parameter summary, only mention it in the trade-off section).
3. Provide the final state machine and architectural parameters.
"""
        self.strategy_adapter.completion(prompt, force_json=False)
        return self.strategy_adapter.last_response
