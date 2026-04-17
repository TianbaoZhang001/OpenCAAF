from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    CONVERGED = "converged"
    FAILED = "failed"
    PARADOX = "paradox"

class VectorizedGradient(BaseModel):
    """Specific, directional feedback for a task."""
    criterion: str
    score: float = Field(..., ge=0.0, le=1.0)
    instruction: str = Field(..., description="Actionable instruction to correct the deviation.")
    failed_value: Optional[Any] = None
    target_value: Optional[Any] = None

class AtomicTask(BaseModel):
    """The smallest unit of work in OpenCAAF."""
    id: str
    parent_id: Optional[Union[str, List[str]]] = None
    description: str
    context_keys: List[str] = Field(default_factory=list, description="Keys of variables required from parent context.")
    expected_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for the output.")
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    gradients: List[VectorizedGradient] = Field(default_factory=list)
    attempts: int = 0

class DecompositionTree(BaseModel):
    """The computation graph of a high-level request."""
    root_request: str
    nodes: Dict[str, AtomicTask]
    global_state: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TraceEntry(BaseModel):
    step: str
    role: str
    model: str
    system_prompt: str
    user_input: str
    response: str
    usage: Dict[str, int]
    convergence_score: Optional[float] = None

class ExecutionMetrics(BaseModel):
    convergence_history: List[float] = []
    total_duration: float = 0.0
    tokens_used: int = 0
    termination_reason: str = "completed"

class ExecutionTrace(BaseModel):
    domain: str
    user_request: str
    entries: List[TraceEntry] = []
    final_status: str
    total_attempts: int = 0
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
