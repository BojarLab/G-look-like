import networkx as nx


def compute_stats(values, prefix="sasa"):
    """
    Compute statistics (mean, max, sum) for a list of values.

    Args:
        values: List of numerical values
        prefix: String prefix for the stats keys (default: "sasa")

    Returns:
        Dictionary containing the computed statistics with the specified prefix
    """
    if not values:  # Check if list is empty
        return {
            f"{prefix}_mean": None,
            f"{prefix}_max": None,
            f"{prefix}_sum": None
        }

    total_sum = sum(values)  # Changed from 'sum = sum(values)'
    mean_value = total_sum / len(values)
    max_value = max(values)

    return {
        f"{prefix}_mean": mean_value,
        f"{prefix}_max": max_value,
        f"{prefix}_sum": total_sum
    }


def extract_node_attributes(glycan_graph: nx.Graph,
                            matched_nodes: list,
                            mono_only: bool = True) -> dict:

    """
    Extracts attributes from specified nodes in a glycan graph.

    Args:
        glycan_graph (nx.Graph): NetworkX graph representing the glycan.
        matched_nodes (list): List of node IDs to extract attributes from.
        mono_only (bool, optional): Only process monosaccharide nodes (even IDs). Defaults to True.

    Returns:
        dict: Dictionary containing:
            'monosaccharides': List of monosaccharide types
            'sasa_values': List of SASA values
            'flexibility_values': List of flexibility values
            'node_ids': List of processed node IDs
    """
    if not isinstance(glycan_graph, nx.Graph):
        return {
            'monosaccharides': [],
            'sasa_values': [],
            'flexibility_values': [],
            'node_ids': []
        }

    # Filter nodes if mono_only is True
    if mono_only:
        selected_nodes = [node for node in matched_nodes
                          if node in glycan_graph.nodes and node % 2 == 0]
    else:
        selected_nodes = [node for node in matched_nodes
                          if node in glycan_graph.nodes]

    # Initialize result containers
    monosaccharides = []
    sasa_values = []
    flexibility_values = []
    node_ids = []

    # Extract attributes for each node
    for node in selected_nodes:
        try:
            node_data = glycan_graph.nodes[node]
            mono_type = node_data.get('Monosaccharide', '')

            if mono_type:  # Only process valid monosaccharides
                monosaccharides.append(mono_type)
                sasa_values.append(node_data.get('SASA', 0))
                flexibility_values.append(node_data.get('flexibility', 0))
                node_ids.append(node)

        except Exception:
            pass

    return {
        'monosaccharides': monosaccharides,
        'sasa_values': sasa_values,
        'flexibility_values': flexibility_values,
        'node_ids': node_ids
    }