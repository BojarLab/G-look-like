import networkx as nx
from scripts.find import find_matching_motif_nodes, find_motif_mono_nodes, find_node_attr
from scripts.agg import process_node_attributes, aggregate_values
import logging
# Create a logger for this module
logger = logging.getLogger(__name__)

def process_glycan_with_motifs_(glycan_id: str,
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
    motif_nodes, matched_motifs = find_matching_motif_nodes(glycan_id, properties)
    # Get the graph for this glycan
    glycan_graph = flex_data.get(glycan_id)
    mono_nodes = find_motif_mono_nodes(motif_nodes, glycan_graph)


    # Initialize results with per-motif tracking
    per_motif_results = []
    all_monosaccharides = []
    all_sasa_values = []
    all_flexibility_values = []

    # Process each set of matched nodes with its corresponding motif
    for motif_nodes, motif in zip(motif_nodes, matched_motifs):
        # Extract attributes for this motif
        attributes = find_node_attr(glycan_graph, motif_nodes, True)

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


def process_glycan_with_motifs(glycan_id, properties, flex_data):
    """
    Processes a glycan and maintains separation between different motifs.

    Args:
        glycan_id (str): Identifier of the glycan to analyze.
        properties (dict): Dictionary containing motifs and terminal residues.
        flex_data (dict): Mapping from glycan IDs to their NetworkX graph representations.

    Returns:
        dict: Processed glycan attributes with aggregations
    """
    # Find matching nodes for each motif
    motif_nodes, matched_motifs = find_matching_motif_nodes(glycan_id, properties)

    # If no motifs matched, return empty results
    if not matched_motifs:
        return {
            'per_motif': [],
            'glycan': {
                'mono': '',
                'sasa': 0.0,
                'flex': 0.0
            },
            'matched_motifs': []
        }

    # Get the graph for this glycan
    glycan_graph = flex_data.get(glycan_id)
    if not glycan_graph:
        logger.error(f"Graph not found for glycan: {glycan_id}")
        return {
            'per_motif': [],
            'glycan': {
                'mono': '',
                'sasa': 0.0,
                'flex': 0.0
            },
            'matched_motifs': []
        }

    # Filter for monosaccharide nodes
    mono_nodes = find_motif_mono_nodes(motif_nodes, glycan_graph)

    #if mono_nodes

    # Extract attributes for these nodes
    node_attrs = find_node_attr(glycan_graph, mono_nodes)

    # Process attributes with aggregation
    attrs_result = process_node_attributes(
        node_attrs,
        agg_within='sum',  # Sum within motifs
        agg_across='max'  # Max across motifs
    )

    # Combine results
    result = {
        'per_motif': [],
        'glycan': attrs_result['glycan'],
        'matched_motifs': matched_motifs
    }

    # Add per-motif details with motif names
    for i, (motif, attrs) in enumerate(zip(matched_motifs, attrs_result['per_motif'])):
        result['per_motif'].append({
            'motif': motif,
            'mono': attrs['mono'],
            'sasa': attrs['sasa'],
            'flex': attrs['flex']
        })

    return result
