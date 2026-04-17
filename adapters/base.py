from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMAdapter(ABC):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_prompt: str = ""
        self.last_response: str = ""
        self.last_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @abstractmethod
    def completion(self, prompt: str, schema: Optional[Dict[str, Any]] = None, force_json: bool = False) -> Dict[str, Any]:
        """Unified method for structured output generation."""
        pass
    
    def reset_usage(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
