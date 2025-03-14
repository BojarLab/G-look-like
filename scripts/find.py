import pandas as pd
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.graph import glycan_to_nxGraph
from collections import defaultdict
import logging
import numpy as np
import networkx as nx
logger = logging.getLogger(__name__)

def find_matching_glycan(flex_data, glycan):
    """Find the matching glycan in flex_data."""
    for flex_glycan in flex_data.keys():
        if compare_glycans(glycan, flex_glycan):
            return glycan
    return None

def magic(glycan_id: str, properties: dict, flex_data: dict[str, nx.Graph],
          within, btw) -> pd.DataFrame:
    """
    Generate metrics for a glycan based on motif matching.

    Parameters:
    -----------
    glycan_id : str
        The ID of the glycan to analyze
    properties : dict
        Dictionary containing motif and termini information
    flex_data : dict[str, nx.Graph]
        Dictionary mapping glycan IDs to their NetworkX graph representations
    within : function, default=np.nansum
        Function to aggregate values within a match (options: np.nansum, np.nanmean, np.nanmax)
    btw : function, default=np.mean
        Function to aggregate values between matches (options: np.sum, np.mean, np.max)

    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing the metrics if a match is found, None otherwise
    """
    motifs = properties.get("motif", [])
    termini_list = properties.get("termini_list", [])

    for i, motif in enumerate(motifs):
        try:
            # Get the specific termini for this motif
            if i < len(termini_list):
                termini = termini_list[i]
            else:
                print(f"Warning: No termini found for motif {motif}")
                termini = []

            # Pass only the specific termini for this motif
            motif_graph = glycan_to_nxGraph(motif, termini='provided', termini_list=termini)
            glycan_graph = flex_data.get(glycan_id)

            if glycan_graph is None:
                print(f"Warning: No graph found for glycan {glycan_id}")
                continue

            is_present, matches = subgraph_isomorphism(glycan_graph, motif_graph,
                                                       return_matches=True)
            if not is_present:
                continue

            sasa, flex, mono = [], [], []
            for m in matches:
                # Collect monosaccharide information
                mono.append([glycan_graph.nodes()[n].get('Monosaccharide', "") for n in m])

                # Apply the within-match aggregation function
                sasa.append(within([glycan_graph.nodes()[n].get('SASA', np.nan) for n in m]))
                flex.append(within([glycan_graph.nodes()[n].get('flexibility', np.nan) for n in m]))

            # Apply the between-match aggregation function
            final_sasa = btw(sasa)
            final_flex = btw(flex)

            # Convert motif to string if it's a list to avoid unhashable type error
            motif_value = str(motif) if isinstance(motif, list) else motif

            # Create a DataFrame with the results
            magic_df = pd.DataFrame({
                "glycan": glycan_id,
                "SASA": [final_sasa],
                "flexibility": [final_flex],
                "monosaccharides": [mono],
                "motifs": [motif_value],
                # Add information about the aggregation methods used
                "within_method": [within.__name__],
                "between_method": [btw.__name__]
            })

            return magic_df

        except Exception as e:
            print(f"Error processing motif {motif}: {e}")
            continue

    return None
