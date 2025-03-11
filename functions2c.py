import pickle
from collections import defaultdict
from typing import Dict, List, Any
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
import importlib.resources as pkg_resources
import glycontact
import networkx as nx
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.stats import pearsonr, spearmanr



BINDING_DATA_PATH_v3 = 'data/20250216_glycan_binding.csv'
BINDING_DATA_PATH_v2 = 'data/20241206_glycan_binding.csv'  # seq,protein
BINDING_DATA_PATH_v1 = 'data/glycan_binding.csv'  # seq,protein


def generate_metrics_for_glycan(properties: dict, glycan_scores: dict,
                            flex_data: dict[str, nx.Graph]) -> tuple[list[dict], dict]:
    """
    Generates metrics for each glycan by processing them one by one and computing final attributes.
    Also collects glycans with multi-node motifs for aggregation testing.

    Returns:
        tuple: (metric_data, agg_test_glycans)
            - metric_data: List of dictionaries with metrics for each processed glycan
            - agg_test_glycans: Dictionary mapping glycan IDs to their aggregation test data
    """
    print("Starting generate_metrics_for_glycan function")  # Debug print

    def find_matching_glycan(flex_data, glycan):
        """Find the matching glycan in flex_data."""
        for flex_glycan in flex_data.keys():
            if compare_glycans(glycan, flex_glycan):
                return flex_glycan
        return None

    # Initialize result containers
    metric_data = []
    missing_glycans = []
    agg_test_glycans = {}

    # Process each glycan
    for glycan in glycan_scores:
        binding_score = glycan_scores[glycan]
        matched_glycan = find_matching_glycan(flex_data, glycan)

        if not matched_glycan:
            missing_glycans.append(glycan)
            continue

        # Now call the process_glycan_attributes_with_motifs function
        glycan_metrics, agg_test_data = process_glycan_attributes_with_motifs(
            matched_glycan, properties, flex_data,
            node_agg='sum', occ_agg='max'
        )

        if glycan_metrics["monos"]:
            metric_data.append({
                "glycan": glycan,
                "binding_score": binding_score,
                "monosaccharides": glycan_metrics["monos"], #matched monosaccharides
                "motifs": glycan_metrics["found_motifs"],
                "sasa": glycan_metrics["final_sasa"],
                "flexibility": glycan_metrics["final_flex"],
                "has_multi_node_motifs": glycan_metrics["has_multi_nodes"]
            })

            # If this glycan has multi-node motifs, add it to the testing collection
            if agg_test_data["has_multi_nodes"]:  # previously "multi_node_motifs"
                agg_test_glycans[glycan] = {
                    "binding_score": binding_score,
                    "motif_data": agg_test_data["multi_node_parts"]  # previously "motif_parts_data"
                }

    print(f"Processed {len(metric_data)} glycans with metrics.")
    print(f"Found {len(agg_test_glycans)} glycans with multi-node motifs for aggregation testing.")

    # Critical: Return the results
    return metric_data, agg_test_glycans

def process_glycan_attributes_with_motifs(glycan_id, props, flex_data, node_agg='sum', occ_agg='max'):
    """
    Process glycan to compute SASA and flexibility metrics.

    Uses a uniform approach for all cases:
    1. Always aggregate values within each occurrence (Level 1)
    2. Always aggregate across occurrences (Level 2)

    Note: Aggregation functions handle single values appropriately:
    - sum([x]) = x
    - mean([x]) = x
    - max([x]) = x
    """

    def safe_agg(values, method='mean', default=0):
        """Safely aggregate values, handling empty lists."""
        if not values:
            return default
        if method == 'mean':
            return np.mean(values)
        elif method == 'sum':
            return np.sum(values)
        elif method == 'max':
            return np.max(values)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_nodes(glycan_id, motif, termini):
        """Get matched nodes using subgraph isomorphism."""
        try:
            is_present, nodes = subgraph_isomorphism(
                glycan_id, motif, return_matches=True, termini_list=termini
            )
            return nodes if is_present else None
        except Exception as e:
            logger.error(f"Error for glycan {glycan_id} with motif {motif}: {e}")
            return None

    def filter_monos(nodes, graph):
        """Filter to keep only monosaccharide nodes."""
        return [
            [node for node in part if node in graph.nodes and node % 2 == 0]
            for part in nodes
        ]

    def extract_attrs(node_lists, graph):
        """Extract attributes from nodes."""
        all_attrs = []
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

    # Get graph and initialize data structures
    motifs = props.get("motif", [])
    termini = props.get("termini_list", [])
    graph = flex_data.get(glycan_id)

    if not isinstance(graph, nx.Graph):
        logger.warning(f"No valid graph for glycan: {glycan_id}")
        return {}, {"has_multi_nodes": False}

    metrics = {
        "monos": [],
        "sasa_raw": [],
        "flex_raw": [],
        "found_motifs": [],
        "has_multi_nodes": False,
        "final_sasa": 0,
        "final_flex": 0
    }

    test_data = {
        "multi_node_parts": [],
        "has_multi_nodes": False
    }

    # Find and process motifs
    for motif, term in zip(motifs, termini):
        matched = get_nodes(glycan_id, motif, term)
        if not matched:
            continue

        metrics["found_motifs"].append(motif)
        mono_nodes = filter_monos(matched, graph)
        occ_attrs = extract_attrs(mono_nodes, graph)

        if not occ_attrs:
            continue

        # Values after Level 1 aggregation (per occurrence)
        occ_sasa_values = []
        occ_flex_values = []

        # Process each occurrence
        for attrs in occ_attrs:
            monos = attrs.get("monos", [])
            sasa = attrs.get("sasa", [])
            flex = attrs.get("flex", [])

            if not monos:
                continue

            # Store raw values for reference
            metrics["monos"].extend(monos)
            metrics["sasa_raw"].extend(sasa)
            metrics["flex_raw"].extend(flex)

            # Track if we have multi-node motifs (for testing purposes)
            if len(monos) > 1:
                metrics["has_multi_nodes"] = True
                test_data["has_multi_nodes"] = True
                test_data["multi_node_parts"].append({
                    "monos": monos,
                    "sasa": sasa,
                    "flex": flex,
                    "node_count": len(monos)
                })

            # Level 1: Aggregate within occurrence
            # Note: This works for both single-node and multi-node occurrences
            occ_sasa = safe_agg(sasa, node_agg)
            occ_flex = safe_agg(flex, node_agg)
            occ_sasa_values.append(occ_sasa)
            occ_flex_values.append(occ_flex)

        # Level 2: Aggregate across occurrences
        # Note: This works for both single-occurrence and multi-occurrence cases
        if occ_sasa_values:
            metrics["final_sasa"] = safe_agg(occ_sasa_values, occ_agg)
            metrics["final_flex"] = safe_agg(occ_flex_values, occ_agg)

    return metrics, test_data


def metric_df(lectin, properties : dict) -> tuple[pd.DataFrame, dict]:
    """
    Generate a glycan metric DataFrame for a given lectin and its properties.
    Now also returns glycans suitable for aggregation testing.
    """
    # Create necessary directories if they don't exist
    os.makedirs('results/metric_df', exist_ok=True)
    os.makedirs('results/agg_test', exist_ok=True)

    def load_data():
        """Load glycan flexibility and binding data, process graphs, and return results."""

        def load_data_pdb():
            """Load glycan flexibility data from PDB source."""
            with pkg_resources.files(glycontact).joinpath("glycan_graphs.pkl").open("rb") as f:
                return pickle.load(f)

        flex_data = load_data_pdb()
        invalid_graphs = [glycan for glycan in flex_data if not isinstance(flex_data[glycan], nx.Graph)]

        def map_protein_to_target(df_target, df_map):
            """
            Maps protein names to their corresponding targets (sequences) in df_target
            using mapping from df_map.

            Args:
                df_target (pd.DataFrame): DataFrame that needs the target column updated.
                df_map (pd.DataFrame): DataFrame containing the protein-to-target mapping.

            Returns:
                pd.DataFrame: Updated df_target with mapped target values.
            """
            # Create a mapping dictionary {target -> protein}
            target_to_protein = dict(zip(df_map["target"], df_map["protein"]))

            # Apply mapping to create the "protein" column in df_target
            df_target["protein"] = df_target["target"].map(target_to_protein)

            return df_target

        binding_df_v2 = pd.read_csv(BINDING_DATA_PATH_v2)
        binding_df_v3 = pd.read_csv(BINDING_DATA_PATH_v3)

        binding_df = map_protein_to_target(binding_df_v3, binding_df_v2)

        return flex_data, binding_df, invalid_graphs

    def filter_binding_data(binding_df: pd.DataFrame, lectin: str) -> pd.DataFrame:
        """select the lectin."""
        filtered_df = binding_df[binding_df.iloc[:, -1].eq(lectin)]
        filtered_df = filtered_df.dropna(axis=1, how='all')
        return filtered_df

    def get_glycan_scores(filtered_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean binding scores for glycans."""
        lectin_df = filtered_df.drop(columns=["target", "protein"])  # Exclude "protein" and "target" columns
        glycan_scores = lectin_df.mean(axis=0).to_dict()
        return glycan_scores

    flex_data, binding_df, invalid_graphs = load_data()
    lectin_df = filter_binding_data(binding_df, lectin)
    if lectin_df.empty:
        print(f"No binding data found for {lectin}.")
        return pd.DataFrame(), {}

    glycan_scores = get_glycan_scores(lectin_df)
    metric_data, agg_test_glycans = generate_metrics_for_glycan(properties, glycan_scores, flex_data)

    metric_df = pd.DataFrame(metric_data)
    if 'glycan' in metric_df.columns:
        metric_df.set_index('glycan', inplace=True)
    else:
        print(f"⚠️ Warning: 'glycan' column missing for {lectin}. Skipping index setting.")

    # Save to Excel
    metric_df.to_excel(f'results/metric_df/{lectin}_metrics.xlsx', index=True, header=True)

    if agg_test_glycans:
        summary_data = []
        for glycan, data in agg_test_glycans.items():
            summary_data.append({
                'glycan': glycan,
                'binding_score': data['binding_score'],
                'motif_count': len(data['motif_data']),
                'total_parts': sum(len(motif) for motif in data['motif_data'])
            })

        # Save summary to Excel
        agg_test_summary = pd.DataFrame(summary_data)
        agg_test_summary.set_index('glycan', inplace=True)
        agg_test_summary.to_excel(f'results/agg_test/{lectin}_agg_test_candidates.xlsx', index=True)


        print(f"Saved {len(agg_test_glycans)} glycans for aggregation testing.")

    return metric_df, agg_test_glycans


def find_best_combo(df):
    """Find the best aggregation method combination based on correlation strength."""
    # Filter to significant results
    sig_df = df[df['pearson_sig'] | df['spearman_sig']].copy()

    if len(sig_df) == 0:
        # Return a dictionary with default values instead of None
        print("Warning: No significant correlations found")
        return {
            'within': None,
            'across': None,
            'pearson_r': None,
            'spearman_r': None
        }

    # Create an absolute value column for sorting
    sig_df['abs_pearson'] = sig_df['pearson_r'].abs()
    sig_df['abs_spearman'] = sig_df['spearman_r'].abs()
    sig_df['max_corr'] = sig_df[['abs_pearson', 'abs_spearman']].max(axis=1)

    # Get the row with the highest correlation
    best_row = sig_df.loc[sig_df['max_corr'].idxmax()]
    result = best_row.to_dict()

    # Ensure keys match what's expected
    if 'within_method' in result and 'within' not in result:
        result['within'] = result['within_method']
    if 'across_method' in result and 'across' not in result:
        result['across'] = result['across_method']

    return result

def compare_agg_methods(glycans_dict, output_path="results/stats/"):
    """
    Compare different aggregation methods for SASA and flexibility.
    Uses 2 levels of aggregation:
    1. Within motif (combining nodes in a motif)
    2. Across motifs (combining different motifs in a glycan)

    Exports results to Excel files and returns DataFrames.

    Args:
        glycans_dict: Dictionary of glycans with binding scores and motif data
        output_path: Directory to save Excel files

    Returns:
        sasa_df: DataFrame with SASA correlation results
        flex_df: DataFrame with flexibility correlation results
    """
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from pathlib import Path

    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # [All your existing code until line 150 - up to the DataFrame creation and significance indicators]

    # Aggregation methods to test
    agg_methods = ['sum', 'mean', 'max']

    # Results storage
    sasa_results = []
    flex_results = []

    # Loop through each lectin
    for lectin, glycans in glycans_dict.items():
        print(f"Analyzing {lectin} with {len(glycans)} glycans")

        # Try all combinations of aggregation methods
        for within in agg_methods:
            for across in agg_methods:
                # Store correlation data
                sasa_data = {
                    'lectin': lectin,
                    'within': within,
                    'across': across,
                    'pearson_r': None, 'pearson_p': None,
                    'spearman_r': None, 'spearman_p': None
                }

                flex_data = {
                    'lectin': lectin,
                    'within': within,
                    'across': across,
                    'pearson_r': None, 'pearson_p': None,
                    'spearman_r': None, 'spearman_p': None
                }

                # Lists for scores and properties
                scores = []
                sasa_vals = []
                flex_vals = []

                # Process each glycan
                for glycan, data in glycans.items():
                    score = data['binding_score']

                    # Lists for motif values
                    motif_sasa = []
                    motif_flex = []

                    # Process each motif
                    for motif in data['motif_data']:
                        # Extract all SASA and flexibility values from this motif
                        all_sasa = []
                        all_flex = []

                        # Get all values from all parts in this motif
                        for part in motif:
                            # Check if part is a dictionary with SASA/flexibility attributes
                            if isinstance(part, dict):
                                sasa = part.get('SASA', [])
                                flex = part.get('flexibility', [])
                            else:
                                # Skip if part is not a dictionary
                                continue

                            # Add to lists
                            all_sasa.extend(sasa)
                            all_flex.extend(flex)

                        # Skip if no values
                        if not all_sasa or not all_flex:
                            continue

                        # First level aggregation (within motif)
                        if within == 'sum':
                            motif_sasa_val = sum(all_sasa)
                            motif_flex_val = sum(all_flex)
                        elif within == 'mean':
                            motif_sasa_val = sum(all_sasa) / len(all_sasa)
                            motif_flex_val = sum(all_flex) / len(all_flex)
                        elif within == 'max':
                            motif_sasa_val = max(all_sasa)
                            motif_flex_val = max(all_flex)

                        motif_sasa.append(motif_sasa_val)
                        motif_flex.append(motif_flex_val)

                    # Skip if no motifs had values
                    if not motif_sasa or not motif_flex:
                        continue

                    # Second level aggregation (across motifs)
                    if across == 'sum':
                        final_sasa = sum(motif_sasa)
                        final_flex = sum(motif_flex)
                    elif across == 'mean':
                        final_sasa = sum(motif_sasa) / len(motif_sasa)
                        final_flex = sum(motif_flex) / len(motif_flex)
                    elif across == 'max':
                        final_sasa = max(motif_sasa)
                        final_flex = max(motif_flex)

                    # Store values
                    scores.append(score)
                    sasa_vals.append(final_sasa)
                    flex_vals.append(final_flex)

                # Calculate correlations if enough data
                if len(scores) >= 3:
                    # SASA correlations
                    sasa_pearson = pearsonr(sasa_vals, scores)
                    sasa_data['pearson_r'] = sasa_pearson[0]
                    sasa_data['pearson_p'] = sasa_pearson[1]

                    sasa_spearman = spearmanr(sasa_vals, scores)
                    sasa_data['spearman_r'] = sasa_spearman[0]
                    sasa_data['spearman_p'] = sasa_spearman[1]

                    # Flexibility correlations
                    flex_pearson = pearsonr(flex_vals, scores)
                    flex_data['pearson_r'] = flex_pearson[0]
                    flex_data['pearson_p'] = flex_pearson[1]

                    flex_spearman = spearmanr(flex_vals, scores)
                    flex_data['spearman_r'] = flex_spearman[0]
                    flex_data['spearman_p'] = flex_spearman[1]

                    # Print results
                    print(f"  {within} within / {across} across:")
                    print(f"    SASA: Pearson r={sasa_pearson[0]:.4f} (p={sasa_pearson[1]:.4f}), " +
                          f"Spearman r={sasa_spearman[0]:.4f} (p={sasa_spearman[1]:.4f})")
                    print(f"    FLEX: Pearson r={flex_pearson[0]:.4f} (p={flex_pearson[1]:.4f}), " +
                          f"Spearman r={flex_spearman[0]:.4f} (p={flex_spearman[1]:.4f})")

                # Add results
                sasa_results.append(sasa_data)
                flex_results.append(flex_data)

    # Create DataFrames
    sasa_df = pd.DataFrame(sasa_results)
    flex_df = pd.DataFrame(flex_results)

    # Add significance indicators
    sasa_df['pearson_sig'] = sasa_df['pearson_p'] < 0.1
    sasa_df['spearman_sig'] = sasa_df['spearman_p'] < 0.1
    flex_df['pearson_sig'] = flex_df['pearson_p'] < 0.1
    flex_df['spearman_sig'] = flex_df['spearman_p'] < 0.1

    # Export the data to Excel files
    sasa_path = str(Path(output_path) / 'sasa_aggregation_methods.xlsx')
    flex_path = str(Path(output_path) / 'flex_aggregation_methods.xlsx')

    sasa_df.to_excel(sasa_path, index=False)
    flex_df.to_excel(flex_path, index=False)

    print(f"\nSASA aggregation data exported to: {sasa_path}")
    print(f"Flexibility aggregation data exported to: {flex_path}")

    # Safely handle finding the best combo (fixing the NoneType error)
    # Modified approach for finding best combo that handles empty DataFrames
    try:
        best_sasa = find_best_combo(sasa_df)
        if best_sasa is not None:
            print("\nBest SASA aggregation method:")
            print(f"  Within: {best_sasa['within']}, Across: {best_sasa['across']}")
            print(f"  Pearson r: {best_sasa['pearson_r']:.4f}, Spearman r: {best_sasa['spearman_r']:.4f}")
        else:
            print("\nNo significant SASA correlations found")
    except Exception as e:
        print(f"\nCould not determine best SASA method: {str(e)}")

    try:
        best_flex = find_best_combo(flex_df)
        if best_flex is not None:
            print("\nBest flexibility aggregation method:")
            print(f"  Within: {best_flex['within']}, Across: {best_flex['across']}")
            print(f"  Pearson r: {best_flex['pearson_r']:.4f}, Spearman r: {best_flex['spearman_r']:.4f}")
        else:
            print("\nNo significant flexibility correlations found")
    except Exception as e:
        print(f"\nCould not determine best flexibility method: {str(e)}")

    return sasa_df, flex_df