import json
from typing import Dict, Any, List, Optional
from OpenCAAF.schemas.core import AtomicTask, DecompositionTree, TaskStatus
from OpenCAAF.adapters.base import LLMAdapter
from OpenCAAF.harness.engine import HarnessRule

class RADEngine:
    """
    Recursive Atomic Decomposition Engine (v0.2).
    Now guided by Enterprise Harnesses to prevent 'Integration Paradoxes'.
    """
    def __init__(self, adapter: LLMAdapter):
        self.adapter = adapter

    def decompose(self, root_request: str, harness_rules: Optional[List[HarnessRule]] = None) -> DecompositionTree:
        """
        Harness-Guided Decomposition (HGD).
        Uses assets to identify coupling and paradoxes during the planning phase.
        """
        harness_context = ""
        if harness_rules:
            harness_context = "\n[MANDATORY ENTERPRISE CONSTRAINTS]\n" + \
                              json.dumps([r.model_dump() for r in harness_rules], indent=2)

        prompt = f"""
        [ROLE: SYSTEMS ARCHITECT]
        Decompose this engineering request into a set of DEPENDENT ATOMIC TASKS.
        {harness_context}

        [ROOT REQUEST]
        {root_request}

        [ARCHITECTURAL RULES]
        1. STRUCTURAL DECOMPOSITION ONLY: Your ONLY job is to create a logical sequence of execution nodes to generate the required variables. DO NOT attempt to evaluate domain constraints, solve mathematical equations, or detect paradoxes yourself.
        2. DEPENDENCY MAPPING: Clearly define 'parent_id' for nodes that rely on previous decisions to form a Directed Acyclic Graph (DAG).
        3. CONTRACT DEFINITION: Each node must have an 'expected_schema' that serves as a binding contract. Ensure the final required outputs are covered by the nodes.

        [OUTPUT FORMAT]
        Return ONLY a JSON object:
        {{
          "root_request": "...",
          "nodes": {{
            "node_id": {{
              "id": "node_id",
              "parent_id": "...",
              "description": "...",
              "context_keys": ["required_variable_names"],
              "expected_schema": {{ ... }}
            }}
          }}
        }}
        """

        raw_tree = self.adapter.completion(prompt, force_json=True)
        
        # Robust parsing logic
        tree_data = raw_tree.get("tree", raw_tree) if isinstance(raw_tree, dict) else raw_tree
        
        nodes = {k: AtomicTask(**v) for k, v in tree_data["nodes"].items()}
        return DecompositionTree(
            root_request=root_request,
            nodes=nodes
        )

    def execute_node(self, task: AtomicTask, tree: DecompositionTree, resolved_context: Dict[str, Any], active_rules: Optional[List[HarnessRule]] = None) -> Dict[str, Any]:
        """
        Contractual Execution: Executor is bound by a specific Harness Clause.
        """
        harness_clause = ""
        if active_rules:
            harness_clause = "\n[ACTIVE HARNESS CONSTRAINTS YOU MUST SATISFY]\n" + \
                             json.dumps([r.model_dump() for r in active_rules], indent=2) + "\n"
        
        override_clause = ""
        if "[STRATEGIC OVERRIDE]" in tree.root_request:
            # Extract just the override part to avoid flooding context
            override_text = tree.root_request[tree.root_request.find("[STRATEGIC OVERRIDE]"):]
            override_clause = f"\n[STRATEGIC SYSTEM OVERRIDE ACTIVE]\n{override_text}\n"
            
        # VERY IMPORTANT: Inject original request so the executor knows the physics rules and formulas
        original_request_context = f"\n[ORIGINAL SYSTEM RULES & FORMULAS]\n{tree.root_request}\n"

        prompt = f"""
        [ROLE: DOMAIN_EXPERT]
        [TASK] {task.description}
        {harness_clause}
        {original_request_context}
        {override_clause}
        [ISOLATED CONTEXT]
        {json.dumps(resolved_context, indent=2)}
        
        [INSTRUCTIONS]
        Execute the task. 
        1. You MUST fulfill the binding contract clause if provided. 
        2. If a STRATEGIC SYSTEM OVERRIDE is active, you MUST respect its new boundaries and use them to calculate your output, overriding any older conflicting limits.
        3. You must find a target value that satisfies ALL remaining constraints.
        
        Return ONLY the data matching this JSON Schema: {json.dumps(task.expected_schema, indent=2)}
        """
        
        return self.adapter.completion(prompt, schema=task.expected_schema)
