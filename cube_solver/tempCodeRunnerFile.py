    queue = [(root, [])]
    visited = set()
    
    while queue and len(queue[0][1]) <= max_solution_depth:
        current_node, path = queue.pop(0)
        
        state_key = tuple(current_node.state.cpu().numpy().flatten())
        if state_key in visited:
            continue
        visited.add(state_key)
        
        if not current_node.children_nodes:
            current_node.expand()
        
        for i, child in enumerate(current_node.children_nodes):
            new_path = path + [i]
            
            if child.is_solved:
                return new_path, True
            
            if len(new_path) < max_solution_depth:
                queue.append((child, new_path))