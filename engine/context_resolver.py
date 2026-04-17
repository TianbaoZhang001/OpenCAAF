from typing import Dict, Any, Optional
from ..schemas.core import AtomicTask, DecompositionTree

class ContextResolver:
    """
    Implements 'Topological Scoping' (Soft Isolation).
    Instead of a hard firewall, it attempts to fetch locally scoped keys first.
    If explicitly needed or missing, it can traverse the decomposition tree upwards.
    """
    
    @staticmethod
    def resolve_context(task: AtomicTask, tree: DecompositionTree) -> Dict[str, Any]:
        """
        Resolves the minimal necessary context for a given task.
        """
        resolved = {}
        
        # 1. Base Resolution (Explicitly declared scope)
        for key in task.context_keys:
            if key in tree.global_state:
                resolved[key] = tree.global_state[key]
            else:
                # 2. Upward Traversal Logic (The 'Lens' approach)
                # If a key is missing from global state but exists in a parent's output,
                # traverse up the parent chain to find it.
                val = ContextResolver._traverse_upwards(task, key, tree)
                if val is not None:
                    resolved[key] = val
                    
        # 3. Inject explicit gradients if the task failed previously
        if task.gradients:
            resolved["__LATEST_FEEDBACK_GRADIENTS__"] = [
                g.model_dump() for g in task.gradients
            ]
            
        return resolved

    @staticmethod
    def _traverse_upwards(task: AtomicTask, target_key: str, tree: DecompositionTree) -> Optional[Any]:
        """Recursively walk up the tree to find missing variables in parent outputs."""
        if not task.parent_id:
            return None
            
        queue = task.parent_id if isinstance(task.parent_id, list) else [task.parent_id]
        visited = set()
        
        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            
            parent_node = tree.nodes.get(curr_id)
            if not parent_node:
                continue
                
            # Check if parent has successfully generated the result containing the key
            if parent_node.result and target_key in parent_node.result:
                return parent_node.result[target_key]
                
            # Move up
            if parent_node.parent_id:
                parents = parent_node.parent_id if isinstance(parent_node.parent_id, list) else [parent_node.parent_id]
                queue.extend(parents)
            
        return None
