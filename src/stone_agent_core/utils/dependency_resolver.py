from typing import Dict, List, Set, Optional, TypeVar, Generic
from collections import defaultdict, deque

T = TypeVar('T')

class DependencyCycleError(Exception):
    """Raised when a circular dependency is detected in the module graph."""
    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")

def resolve_dependencies(dependencies: Dict[str, List[str]]) -> List[str]:
    """
    Resolve module dependencies using topological sort.
    
    Args:
        dependencies: Dictionary mapping module names to their dependencies
        
    Returns:
        List of module names in execution order
        
    Raises:
        DependencyCycleError: If a circular dependency is detected
    """
    # Build the graph and in-degree count
    graph = {module: set() for module in dependencies}
    in_degree = {module: 0 for module in dependencies}
    
    for module, deps in dependencies.items():
        for dep in deps:
            if dep in graph:
                graph[dep].add(module)
                in_degree[module] += 1
    
    # Initialize queue with nodes having no incoming edges
    queue = deque([module for module, degree in in_degree.items() if degree == 0])
    result = []
    
    # Process nodes
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(result) != len(dependencies):
        # Find the cycle
        visited = set()
        path = []
        cycle = []
        
        def visit(node):
            if node in visited:
                return
            if node in path:
                cycle.extend(path[path.index(node):] + [node])
                return
            path.append(node)
            for neighbor in graph.get(node, []):
                if not cycle:
                    visit(neighbor)
            path.pop()
            visited.add(node)
        
        for node in dependencies:
            if not cycle:
                visit(node)
        
        raise DependencyCycleError(cycle)
    
    return result

def validate_dependencies(modules: Dict[str, List[str]]) -> None:
    """
    Validate module dependencies.
    
    Args:
        modules: Dictionary mapping module names to their dependencies
        
    Raises:
        ValueError: If any dependency is invalid
        DependencyCycleError: If a circular dependency is detected
    """
    # Check for missing dependencies
    all_modules = set(modules.keys())
    for module, deps in modules.items():
        missing = [d for d in deps if d not in all_modules]
        if missing:
            raise ValueError(f"Module '{module}' has missing dependencies: {', '.join(missing)}")
    
    # Check for cycles
    resolve_dependencies(modules)