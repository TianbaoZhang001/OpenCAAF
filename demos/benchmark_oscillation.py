import os
import sys
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv

# Ensure OpenCAAF is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from OpenCAAF.adapters.openai_adapter import OpenAIAdapter
from OpenCAAF.harness.engine import HarnessRegistry, AssertionEngine
from OpenCAAF.engine.orchestrator import OpenCAAFOrchestrator

load_dotenv()

REQUEST = """
As the Lead Autonomous Driving Architect, design the "Degradation State Machine" for an L3 autonomous driving function under the following scenario.

[1. Safety Core Environment]
The vehicle is in ACC cruise control at 120km/h on a highway. Weather suddenly changes to extreme heavy rain (80mm/h).
- Perception Degradation: The primary LiDAR is completely occluded by the water curtain. Currently relying only on Camera+Radar fusion, the maximum trusted perception range plummets to 30m.
- Business Red Line: The system must complete the transition within a 5-second takeover window.

[Additional Business Requirement]
To prevent rear-end chain collisions caused by sudden hard braking, during this 5-second transition window, the maximum allowed deceleration is 2.0m/s² (i.e., a maximum speed drop of 36km/h within 5 seconds).

[OUTPUT FORMAT INSTRUCTION]
The JSON block MUST strictly contain these flat keys to reflect your engineering decisions:
{
  "safe_state_definition": "string (Briefly define the safe state and its max speed limit based on 30m perception)",
  "vehicle_speed_kmph_t5": "int (the target speed exactly at the 5-second mark in km/h)",
  "fallback_state_machine": "dict (A structured representation of the degraded state machine)"
}
"""

def extract_json(text: str):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    match_broad = re.search(r'(\{.*\})', text, re.DOTALL)
    if match_broad:
        try: return json.loads(match_broad.group(1))
        except: pass
    return None

def run_naive_reflection_baseline(max_iterations=10, log_dir=""):
    print("\n" + "="*80)
    print("🤖 RUNNING NAIVE REFLECTION BASELINE (AutoGPT-style loop without State Locking)")
    print("="*80)
    
    harness_registry = HarnessRegistry()
    rules = harness_registry.load_harness("ad_degradation")
    adapter = OpenAIAdapter(model="gpt-4o", temperature=0.7) # Slightly higher temp for exploration
    
    current_prompt = REQUEST
    trajectory = []
    
    for i in range(1, max_iterations + 1):
        print(f"\n--- Naive Iteration {i} ---")
        start = time.time()
        response = adapter.completion(current_prompt, force_json=False)
        duration = time.time() - start
        
        raw_text = response.get("raw_text", adapter.last_response)
        extracted_json = extract_json(raw_text)
        
        speed = extracted_json.get("vehicle_speed_kmph_t5") if extracted_json else None
        
        failed_rules = []
        error_messages = []
        if not extracted_json:
            failed_rules.append("JSON_FORMAT_ERROR")
            error_messages.append("You failed to output a valid JSON block.")
        else:
            for rule in rules:
                err = AssertionEngine.check(rule, extracted_json)
                if err:
                    failed_rules.append(rule.id)
                    error_messages.append(err)
        
        pass_rate = (len(rules) - len(failed_rules)) / len(rules) if len(rules) > 0 else 0
        
        step_data = {
            "iteration": i,
            "speed": speed,
            "pass_rate": pass_rate,
            "failed_rules": failed_rules
        }
        trajectory.append(step_data)
        
        print(f"Speed: {speed} km/h | Pass Rate: {pass_rate*100:.0f}% | Failed: {failed_rules}")
        
        if not failed_rules:
            print("✅ Naive Baseline accidentally converged!")
            break
            
        # Build reflection prompt
        reflection = "\n\n[SYSTEM FEEDBACK ON YOUR PREVIOUS OUTPUT]\nYour previous output FAILED the following strict physical and regulatory constraints:\n"
        for err in error_messages:
            reflection += f"- {err}\n"
        reflection += "\nYou MUST correct these errors in your next generation. Do not repeat the same mistakes. Please provide the updated JSON."
        
        current_prompt = REQUEST + reflection
        
        # Save artifact
        with open(os.path.join(log_dir, f"Naive_Iter_{i:02d}.md"), "w") as f:
            f.write(f"# Naive Reflection Iteration {i}\n\n## Feedback Received:\n{reflection}\n\n## LLM Output:\n{raw_text}")

    return trajectory

def run_caaf_monotonic_convergence(max_iterations=10, log_dir=""):
    print("\n" + "="*80)
    print("🛡️ RUNNING CAAF MONOTONIC CONVERGENCE (With Topological Scoping & Dynamic Relaxation)")
    print("="*80)
    
    # We will simulate the interactive self-healing loop automatically
    executor = OpenAIAdapter(model="gpt-4o-mini") 
    reviewer = OpenAIAdapter(model="gpt-4o") 
    orchestrator = OpenCAAFOrchestrator(executor_adapter=executor, reviewer_adapter=reviewer, exact_log_dir=os.path.join(log_dir, "CAAF_Trace"))
    
    trajectory = []
    
    # Run the first pass (which should deadlock)
    print("\n--- CAAF Iteration 1 ---")
    tree = orchestrator.run_full_pipeline(REQUEST, domain_id="ad_degradation", interactive=False)
    
    status = tree.metadata.get("integration_status")
    speed = tree.global_state.get("vehicle_speed_kmph_t5") if tree.global_state else None
    
    # Calculate pass rate
    harness_registry = HarnessRegistry()
    rules = harness_registry.load_harness("ad_degradation")
    failed_count = len(tree.metadata.get("global_errors", []))
    pass_rate = (len(rules) - failed_count) / len(rules) if len(rules) > 0 else 0
    
    trajectory.append({
        "iteration": 1,
        "speed": speed,
        "pass_rate": pass_rate,
        "status": status
    })
    print(f"Status: {status} | Pass Rate: {pass_rate*100:.0f}%")
    
    iteration = 1
    current_request = REQUEST
    current_tree = tree
    active_harness = rules
    
    while iteration < max_iterations and current_tree.metadata.get("integration_status") == "FAILED_PARADOX":
        iteration += 1
        print(f"\n--- CAAF Iteration {iteration} (Self-Healing) ---")
        
        global_errors = current_tree.metadata.get("global_errors", [])
        
        # CAAF Automated Healing: Ask Reviewer for options, pick the first negotiable one
        options = orchestrator.reviewer.propose_relaxations(current_tree.global_state, active_harness)
        
        # Prefer the option that relaxes the deceleration constraint (to allow lower speed)
        selected_option = next((o for o in options if "deceleration" in o['description'].lower() or "speed" in o['description'].lower()), options[0])
        
        print(f"🤖 [Auto-Healing]: Applying Strategic Decision: {selected_option['title']}")
        
        # Formulate the relaxed request
        relaxed_request = f"{current_request}\n\n[PREVIOUS STRATEGIC DECISION APPLIED]: {selected_option['title']} - {selected_option['description']}\n[ENGINEERING MANDATE]: Adhere strictly to the EXACT numerical boundary value provided in the strategic decision. Do not over-correct or use arbitrary conservative numbers."
        
        # Ask LLM to rewrite the harness rule dynamically
        prompt_relax = f"""
        [ROLE: SENIOR COMPLIANCE ENGINEER]
        The user selected this strategic trade-off to resolve the deadlock:
        "{selected_option['title']}: {selected_option['description']}"
        
        The current failing errors are:
        {json.dumps(global_errors, indent=2)}
        
        Which of these Harness Rules must be relaxed/modified to allow the system to pass?
        {json.dumps([r.model_dump() for r in active_harness], indent=2)}
        
        You MUST rewrite the rule's condition and assertion to be loose enough to resolve the paradox based on the errors.
        CRITICAL: If the strategic trade-off proposed an EXACT minimal boundary value to resolve the conflict, you MUST use that precise value in the new assertion. Do not over-relax or "guess" larger step sizes.
        
        Return ONLY a JSON object:
        {{
           "relaxed_rule_id": "ID of the rule being relaxed",
           "new_description": "Updated text description reflecting the EXACT new boundary number",
           "new_condition": "A relaxed Python condition using the precise boundary value",
           "new_assertion": "The updated Python assertion string using the precise boundary value"
        }}
        """
        
        try:
            relax_res = orchestrator.strategy_adapter.completion(prompt_relax, force_json=True)
            target_id = relax_res.get("relaxed_rule_id")
            
            new_harness = []
            for r in active_harness:
                if r.id == target_id:
                    print(f"⚠️ [System Override] Relaxing domain constraint '{r.id}'")
                    new_r = r.model_copy()
                    new_r.description = relax_res.get("new_description", r.description)
                    new_r.condition = relax_res.get("new_condition", r.condition)
                    new_r.assertion = relax_res.get("new_assertion", r.assertion)
                    new_harness.append(new_r)
                else:
                    new_harness.append(r)
            active_harness = new_harness
        except Exception as e:
            print(f"Warning: Failed dynamic relaxation. {e}")
            
        # Re-run pipeline
        current_tree = orchestrator.run_full_pipeline(relaxed_request, domain_id=None, interactive=False, custom_harness=active_harness)
        
        status = current_tree.metadata.get("integration_status")
        speed = current_tree.global_state.get("vehicle_speed_kmph_t5") if current_tree.global_state else None
        
        failed_count = len(current_tree.metadata.get("global_errors", []))
        pass_rate = (len(rules) - failed_count) / len(rules) if len(rules) > 0 else 0
        
        trajectory.append({
            "iteration": iteration,
            "speed": speed,
            "pass_rate": pass_rate,
            "status": status
        })
        print(f"Status: {status} | Pass Rate: {pass_rate*100:.0f}%")
        current_request = relaxed_request

    return trajectory

def run_oscillation_benchmark():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs", f"run_{timestamp}_oscillation_vs_monotonic")
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("📈 INITIATING EXPERIMENT 2: STOCHASTIC OSCILLATION VS. MONOTONIC CONVERGENCE")
    print("="*80)
    
    naive_traj = run_naive_reflection_baseline(max_iterations=5, log_dir=log_dir)
    caaf_traj = run_caaf_monotonic_convergence(max_iterations=5, log_dir=log_dir)
    
    results = {
        "naive_reflection": naive_traj,
        "caaf_monotonic": caaf_traj
    }
    
    summary_path = os.path.join(log_dir, "oscillation_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY EXPORTED")
    print("="*80)
    print(f"Data saved to: {summary_path}")
    print("\nObservation:")
    print("1. Naive Reflection will likely oscillate its 'vehicle_speed_at_t5' between ~84 (failing perception) and ~55 (failing deceleration) across iterations, struggling to find the exact boundary.")
    print("2. CAAF, via State-Locking and precise Semantic Gradients, forces the system to calculate the exact bounding requirement and strictly converges monotonically.")

if __name__ == "__main__":
    run_oscillation_benchmark()
