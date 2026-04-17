import json
from typing import Dict, Any, List, Optional
from OpenCAAF.schemas.core import AtomicTask, VectorizedGradient, TaskStatus
from OpenCAAF.adapters.base import LLMAdapter
from OpenCAAF.harness.engine import HarnessRule, AssertionEngine
from OpenCAAF.engine.solver import UAISolver

class SemanticReviewer:
    """
    The Hybrid Sensor (v0.3.5).
    Systemic Resolution Engine: Capable of calculating exact relaxation magnitudes 
    to bridge the gap between conflicting physical constraints.
    """
    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter

    def compute_gradients(self, task: AtomicTask, artifact: Dict[str, Any], harness_rules: List[HarnessRule], is_global: bool = False) -> List[VectorizedGradient]:
        gradients = []
        
        # 1. DETERMINISTIC HARD-CHECK
        if is_global:
            active_rules = harness_rules
        else:
            task_outputs = list(task.expected_schema.keys())
            active_rules = [r for r in harness_rules if r.target_field in task_outputs]
        
        for rule in active_rules:
            err_report = AssertionEngine.check(rule, artifact)
            if err_report:
                gradients.append(VectorizedGradient(
                    criterion=rule.id,
                    score=0.0,
                    instruction=f"HARD CONSTRAINT VIOLATION: {rule.description}",
                    failed_value=str(err_report.get("actual_value", "logic_fault")),
                    target_value=rule.condition
                ))

        # 2. SEMANTIC ANALYSIS
        harness_text = json.dumps([r.model_dump() for r in active_rules], indent=2)
        prompt = f"""
        [ROLE] Compliance Auditor.
        [CONTRACT] {harness_text}
        [ARTIFACT] {json.dumps(artifact, indent=2)}
        [INSTRUCTION] Output JSON gradients for any remaining deviations. Return [] if 100% PASS.
        """
        try:
            raw = self.adapter.completion(prompt, force_json=True)
            llm_data = raw if isinstance(raw, list) else raw.get("gradients", [])
            for g in llm_data:
                if not any(eg.criterion == g['criterion'] for eg in gradients):
                    gradients.append(VectorizedGradient(**g))
        except: pass
        return gradients

    def propose_relaxations(self, global_artifact: Dict[str, Any], violated_rules: List[HarnessRule], domain_id: str = "ad_degradation") -> List[Dict[str, str]]:
        """
        STRATEGIC NEGOTIATION: Generate human-interpretable trade-offs.
        Instruction: Calculate exact boundary values to bridge the gap between paradoxes.
        """
        # Call the external MCP Tool (Deterministic Solver)
        solver_hints = UAISolver.analyze_domain_paradox(domain_id, global_artifact)

        prompt = f"""
        [ROLE: CHIEF SYSTEMS ARCHITECT]
        The engineering system has encountered a SYSTEM DEADLOCK. Two or more constraints are physically exclusive.
        
        [VIOLATED RULES]
        {json.dumps([r.model_dump() for r in violated_rules], indent=2)}
        
        [CURRENT SYSTEM STATE]
        {json.dumps(global_artifact, indent=2)}
        
        {solver_hints}
        
        [TASK]
        Propose strategic trade-offs using the exact RECOMMENDED_TARGET values provided by the UAI tool. Do not do any math yourself. Your description must conclude with the instruction: "Set [variable] to exactly [RECOMMENDED_TARGET]. 
        
        [OUTPUT FORMAT]
        Return a JSON object:
        {{
          "options": [
            {{
              "id": "A",
              "title": "Short title",
              "description": "To satisfy [Anchor], the safe limit is [UAI boundary]. To reach it, we must relax [Relaxed Rule] to [safe target]. Set [Variable to relax] to exactly [safe target].",
              "impact": "Pros/cons of this choice"
            }}
          ]
        }}
        """
        
        response = self.adapter.completion(prompt, force_json=True)
        print(f"reviwer prompt: ", prompt)
        print(f"reviwer response: ", response)
        return response.get("options", [])
