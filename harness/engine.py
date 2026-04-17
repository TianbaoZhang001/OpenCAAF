from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import yaml
import os

class HarnessRule(BaseModel):
    """A deterministic constraint rule stored as an enterprise asset."""
    id: str
    description: str
    target_field: Optional[str] = None
    condition: str  # Human readable condition
    assertion: Optional[str] = None  # Executable Python expression (e.g. "input['val'] < 500")
    severity: str = "CRITICAL"

class AssertionEngine:
    """
    The 'Hard-Check' engine for OpenCAAF.
    Executes irrefutable logic defined in the Harness.
    """
    @staticmethod
    def check(rule: HarnessRule, artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Returns a dict with error and actual value if failed, else None.
        """
        if not rule.assertion:
            return None
            
        try:
            # 1. EVALUATE THE TRUTH (Full Assertion)
            is_valid = eval(rule.assertion, {"__builtins__": {}}, {"input": artifact})
            
            if not is_valid:
                # 2. EXTRACT ACTUAL VALUE (Left side of the operator)
                # Split by common operators to isolate the metric being tested
                left_side = rule.assertion.split('<=')[0].split('<')[0].split('==')[0].split('>=')[0].split('>')[0].strip()
                try:
                    actual_value = eval(left_side, {"__builtins__": {}}, {"input": artifact})
                except:
                    actual_value = "evaluation_error"
                
                return {
                    "error": f"HARD ASSERTION FAILED: {rule.description}",
                    "actual_value": actual_value
                }
        except Exception as e:
            # If the full assertion crashes (e.g. missing keys), it's a failure
            return {"error": f"SYSTEM ERROR during assertion: {str(e)}", "actual_value": "missing_data"}
            
        return None

class HarnessRegistry:
    def __init__(self, base_path: str = None):
        if base_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(current_dir, "data")
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def load_harness(self, domain_id: str) -> List[HarnessRule]:
        path = os.path.join(self.base_path, f"{domain_id}.yaml")
        if not os.path.exists(path):
            return []
            
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return [HarnessRule(**r) for r in data.get("rules", [])]
