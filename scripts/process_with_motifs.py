import networkx as nx
from scripts.match_glycan import find_matching_nodes
from scripts.compute_attr import extract_node_attributes

def process_glycan_with_motifs(glycan_id: str,
                               properties: dict,
                               flex_data: dict[str, nx.Graph]) -> dict:
    """
    Processes a glycan and maintains separation between different motifs.

    Args:
        glycan_id (str): Identifier of the glycan to analyze.
        properties (dict): Dictionary containing motifs and terminal residues.
        flex_data (dict): Mapping from glycan IDs to their NetworkX graph representations.

    Returns:
        dict: {
            'per_motif': List of dictionaries, each containing:
                'motif': The matched motif
                'monosaccharides': List of monosaccharide types for this motif
                'sasa_values': SASA values for this motif
                'flexibility_values': Flexibility values for this motif
            'combined': Dictionary with combined results across all motifs
            'matched_motifs': List of all matched motifs
        }
    """

    # Find matching nodes for each motif
    all_matched_nodes, matched_motifs = find_matching_nodes(glycan_id, properties)

    # Get the graph for this glycan
    glycan_graph = flex_data.get(glycan_id)

    # Initialize results with per-motif tracking
    per_motif_results = []
    all_monosaccharides = []
    all_sasa_values = []
    all_flexibility_values = []

    # Process each set of matched nodes with its corresponding motif
    for matched_nodes, motif in zip(all_matched_nodes, matched_motifs):
        # Extract attributes for this motif
        attributes = extract_node_attributes(glycan_graph, matched_nodes, True)

        # Store results for this specific motif
        per_motif_results.append({
            'motif': motif,
            'monosaccharides': attributes['monosaccharides'],
            'sasa_values': attributes['sasa_values'],
            'flexibility_values': attributes['flexibility_values'],
            'node_ids': attributes['node_ids']
        })

        # Also maintain combined results
        all_monosaccharides.extend(attributes['monosaccharides'])
        all_sasa_values.extend(attributes['sasa_values'])
        all_flexibility_values.extend(attributes['flexibility_values'])

    # Return both per-motif and combined results
    return {
        'per_motif': per_motif_results,
        'combined': {
            'monosaccharides': all_monosaccharides,
            'sasa_values': all_sasa_values,
            'flexibility_values': all_flexibility_values
        },
        'matched_motifs': matched_motifs
    }