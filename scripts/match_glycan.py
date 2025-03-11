from glycowork.motif.graph import compare_glycans, subgraph_isomorphism

def find_matching_glycan(flex_data, glycan):
    """Find the matching glycan in flex_data."""
    for flex_glycan in flex_data.keys():
        if compare_glycans(glycan, flex_glycan):
            return glycan
    return None


def find_matching_nodes(glycan_id: str, properties: dict) -> tuple:
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
    all_matched_nodes = []
    matched_motifs = []

    # Extract motifs and their terminal residues
    motifs = properties.get("motif", [])
    termini_list = properties.get("termini_list", [])

    # Process each motif with its corresponding termini
    for motif, termini in zip(motifs, termini_list):
        try:
            # Check if motif is present in the glycan
            is_present, matched_nodes = subgraph_isomorphism(
                glycan_id, motif,
                return_matches=True,
                termini_list=termini
            )

            if not is_present:
                continue

        except Exception:
            continue

        # Record the successful motif match
        matched_motifs.append(motif)

        # Flatten matched nodes if needed and convert to integers
        if matched_nodes and isinstance(matched_nodes[0], list):
            matched_nodes = [node for sublist in matched_nodes for node in sublist]
        matched_nodes = [int(node) for node in matched_nodes]

        # Add these nodes to our collection
        all_matched_nodes.append(matched_nodes)

    return all_matched_nodes, matched_motifs