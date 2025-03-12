from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def find_matching_glycan(flex_data, glycan):
    """Find the matching glycan in flex_data."""
    for flex_glycan in flex_data.keys():
        if compare_glycans(glycan, flex_glycan):
            return glycan
    return None


def find_matching_motif_nodes(glycan_id: str, properties: dict) -> tuple:
    """
    Identifies nodes in a glycan that match specified binding motifs.

    Args:
        glycan_id (str): Identifier of the glycan to analyze.
        properties (dict): Dictionary containing motifs and terminal residues.

    Returns:
        tuple: (
            list[list[int]]: Lists of matched nodes for each successful motif,
            list[str]: Successfully matched motifs
        )
    """
    # Initialize result containers
    motif_nodes = []
    matched_motifs = []

    # Extract motifs and their terminal residues
    motifs = properties.get("motif", [])
    termini_list = properties.get("termini_list", [])

    # Process each motif with its corresponding termini
    for motif, termini in zip(motifs, termini_list):
        try:
            # Check if motif is present in the glycan
            is_present, motif_nodes = subgraph_isomorphism(
                glycan_id, motif,
                return_matches=True,
                termini_list=termini)
            if not is_present:
                continue
        except Exception:
            continue

        matched_motifs.append(motif)

    return  motif_nodes, matched_motifs


def find_motif_mono_nodes(nodes, graph):
    """Filter to keep only monosaccharide nodes.
    Nested structure preserved.
    matched_motif= "Fuc(a1-2)GalNAc", Fuc(a1-2)GalNAc"
    motif_nodes = [[0, 1, 2], [4, 5, 6]]
    mono_nodes= [[0,2], [4,6]]
     """
    return [
        [node for node in part if node in graph.nodes and node % 2 == 0]
        for part in nodes
    ]


def find_node_attr(graph, node_lists):
    """Extract attributes from nodes."""
    all_attrs = []

    # Check if node_lists is a flat list of integers or a nested list
    if node_lists and not isinstance(node_lists[0], list):
        # If it's a flat list, wrap it in another list to make it nested
        node_lists = [node_lists]

    for part in node_lists:
        attrs = defaultdict(list)
        for node in part:
            try:
                node_attr = graph.nodes[node]
                mono = node_attr.get('Monosaccharide', "")
                if mono:
                    attrs["monos"].append(mono)
                    attrs["sasa"].append(node_attr.get("SASA", 0))
                    attrs["flex"].append(node_attr.get("flexibility", 0))
            except Exception as e:
                logger.error(f"Error extracting attributes: {e}")
        all_attrs.append(dict(attrs))
    return all_attrs