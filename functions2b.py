
import pickle
from typing import Dict
import numpy as np
import pandas as pd
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.processing import get_class
import sys
from networkx.classes import Graph
import importlib.resources as pkg_resources
import glycontact

import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import seaborn as sns
import networkx as nx
import os
from scipy.stats import pearsonr, spearmanr

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from scipy.cluster.hierarchy import dendrogram, linkage



BINDING_DATA_PATH_v3 = 'data/20250216_glycan_binding.csv'
BINDING_DATA_PATH_v2 = 'data/20241206_glycan_binding.csv'  # seq,protein
BINDING_DATA_PATH_v1 = 'data/glycan_binding.csv'  # seq,protein



def process_glycan_with_motifs(matched_glycan, properties, flex_data,
                               mono_agg_method='sum', motif_agg_method='max'):
    """
    Processes a glycan structure to identify binding motifs and compute metrics.
    Now also identifies which glycans have motifs with multiple nodes for aggregation testing.

    Args:
        matched_glycan: Glycan identifier
        properties: Contains motifs and termini lists
        flex_data: Maps glycans to their corresponding graphs
        mono_agg_method: How to aggregate values within a single motif ('sum', 'mean', 'max')
        motif_agg_method: How to aggregate across different motifs ('max', 'mean', 'sum')

    Returns:
        Tuple of:
        - Dictionary with computed metrics and extracted attributes
        - Dictionary with aggregation test data for glycans with multi-node motifs
    """


    def safe_aggregate(values, method='mean', default=0):
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
            raise ValueError(f"Unknown aggregation method: {method}")

    def get_matched_nodes(matched_glycan, motif, termini):
        """Identifies matched nodes using subgraph isomorphism."""
        try:
            is_present, matched_nodes = subgraph_isomorphism(
                matched_glycan, motif, return_matches=True, termini_list=termini
            )
            return matched_nodes if is_present else None
        except Exception as e:
            print(f"Subgraph isomorphism error for glycan {matched_glycan} with motif {motif}: {e}")
            return None

    def filter_monosaccharides(matched_nodes, graph):
        """
        Filters out non-monosaccharides (only even-indexed nodes).
        Preserves the nested structure: [[0,2], [4,6]] represents different parts
        of the same motif, while [0,2,4,6] would be flattened.
        """
        return [
            [node for node in motif_part if node in graph.nodes and node % 2 == 0]
            for motif_part in matched_nodes
        ]

    def extract_node_attributes(node_lists, graph):
        """
        Extracts attributes from graph nodes.
        Preserves the nested structure where each sublist is a part of a motif.
        """
        # Initialize with nested structure
        all_attributes = []

        # Process each part of the motif separately
        for motif_part in node_lists:
            part_attributes = {
                "SASA": [],
                "flexibility": [],
                "matched_monosaccharides": []
            }

            for node in motif_part:
                try:
                    node_attr = graph.nodes[node]
                    monosaccharide = node_attr.get('Monosaccharide', "")
                    if monosaccharide:
                        part_attributes["matched_monosaccharides"].append(monosaccharide)
                        part_attributes["SASA"].append(node_attr.get("SASA", 0))
                        part_attributes["flexibility"].append(node_attr.get("flexibility", 0))
                except Exception as e:
                    print(f"Error extracting attributes for node {node}: {e}")

            all_attributes.append(part_attributes)

        return all_attributes

    # Get motifs and termini from properties
    motifs = properties.get("motif", [])
    termini_list = properties.get("termini_list", [])

    # Lists to store results per glycan
    all_matched_monosaccharides = []  # For tracking all found monosaccharides
    all_sasa = []  # For tracking all SASA values
    all_flexibility = []  # For tracking all flexibility values
    found_motifs = []  # Motifs that were successfully found

    # For storing aggregated values per motif
    motif_sasa_values = []
    motif_flex_values = []

    # NEW: Track which motifs have multiple nodes (for aggregation testing)
    has_multi_node_motifs = False
    motif_node_counts = []  # To track number of nodes in each motif part

    # NEW: Store raw data for aggregation testing
    agg_test_data = {
        "motif_parts_data": [],  # Will store raw data for each motif part
        "multi_node_motifs": False  # Flag indicating if any multi-node motifs exist
    }

    # Process each motif
    for motif_idx, (motif, termini) in enumerate(zip(motifs, termini_list)):
        # Get matched nodes from the graph
        matched_nodes = get_matched_nodes(matched_glycan, motif, termini)
        if matched_nodes is None:
            continue

        found_motifs.append(motif)

        # Get the corresponding graph
        graph = flex_data.get(matched_glycan)
        if not isinstance(graph, nx.Graph):
            print(f"No valid graph found for glycan: {matched_glycan}")
            continue

        # Filter to get only monosaccharides, preserving motif parts structure
        mono_nodes = filter_monosaccharides(matched_nodes, graph)

        # Extract attributes for each part, preserving nested structure
        motif_parts_attributes = extract_node_attributes(mono_nodes, graph)

        # Skip if no valid parts were found
        if not motif_parts_attributes:
            continue

        # NEW: Track motif parts with multiple nodes for aggregation testing
        motif_test_data = []

        # First level aggregation - aggregate within each motif part
        part_sasa_values = []
        part_flex_values = []

        for part_attr in motif_parts_attributes:
            # Add all individual monosaccharides to the overall lists
            all_matched_monosaccharides.extend(part_attr["matched_monosaccharides"])
            all_sasa.extend(part_attr["SASA"])
            all_flexibility.extend(part_attr["flexibility"])

            # NEW: Check if this motif part has multiple nodes
            node_count = len(part_attr["matched_monosaccharides"])
            motif_node_counts.append(node_count)

            if node_count > 1:
                has_multi_node_motifs = True

                # NEW: Store this motif part data for aggregation testing
                motif_test_data.append({
                    "monosaccharides": part_attr["matched_monosaccharides"],
                    "SASA": part_attr["SASA"],
                    "flexibility": part_attr["flexibility"],
                    "node_count": node_count
                })

            # Aggregate values within each part of the motif
            if part_attr["matched_monosaccharides"]:  # Only aggregate if we have values
                part_sasa_values.append(safe_aggregate(part_attr["SASA"], mono_agg_method))
                part_flex_values.append(safe_aggregate(part_attr["flexibility"], mono_agg_method))

        # NEW: Add this motif's data to the aggregation test data if it has multi-node parts
        if motif_test_data:
            agg_test_data["motif_parts_data"].append(motif_test_data)

        # Second level aggregation - aggregate across parts for this motif
        # Only if we have at least one valid part
        if part_sasa_values:
            motif_sasa = safe_aggregate(part_sasa_values, mono_agg_method)
            motif_sasa_values.append(motif_sasa)

        if part_flex_values:
            motif_flex = safe_aggregate(part_flex_values, mono_agg_method)
            motif_flex_values.append(motif_flex)

    # Final aggregation across all motifs
    final_sasa = safe_aggregate(motif_sasa_values, motif_agg_method)
    final_flexibility = safe_aggregate(motif_flex_values, motif_agg_method)

    # NEW: Update the flag for multi-node motifs
    agg_test_data["multi_node_motifs"] = has_multi_node_motifs

    # Standard metrics for all glycans
    metrics_dict = {
        "matched_monosaccharides": all_matched_monosaccharides,
        "SASA": all_sasa,
        "flexibility": all_flexibility,
        "found_motifs": found_motifs,
        "final_sasa": final_sasa,
        "final_flexibility": final_flexibility,
        "motif_sasa_values": motif_sasa_values,
        "motif_flex_values": motif_flex_values,
        "has_multi_node_motifs": has_multi_node_motifs  # NEW: Flag for whether this glycan has multi-node motifs
    }

    return metrics_dict, agg_test_data


def generate_metrics_for_glycan(properties: dict, glycan_scores: dict, flex_data: dict[str, nx.Graph]) -> tuple[list[dict], dict]:
    """
    Generates metrics for each glycan by processing them one by one and computing final attributes.
    Also collects glycans with multi-node motifs for aggregation testing.

    Returns:
        tuple: (metric_data, agg_test_glycans)
            - metric_data: List of dictionaries with metrics for each processed glycan
            - agg_test_glycans: Dictionary mapping glycan IDs to their aggregation test data
    """

    def find_matching_glycan(flex_data, glycan):
        """Find the matching glycan in flex_data."""
        for flex_glycan in flex_data.keys():
            if compare_glycans(glycan, flex_glycan):
                return flex_glycan
        return None

    metric_data = []
    missing_glycans = []

    # NEW: Dictionary to collect glycans suitable for aggregation testing
    agg_test_glycans = {}

    for glycan in glycan_scores:
        binding_score = glycan_scores[glycan]
        matched_glycan = find_matching_glycan(flex_data, glycan)

        if not matched_glycan:
            missing_glycans.append(glycan)
            continue

        # NEW: Get both metrics and aggregation test data
        glycan_metrics, agg_test_data = process_glycan_with_motifs(
            matched_glycan, properties, flex_data,
            mono_agg_method='sum', motif_agg_method='max'
        )

        if glycan_metrics["matched_monosaccharides"]:
            metric_data.append({
                "glycan": glycan,
                "binding_score": binding_score,
                "monosaccharides": glycan_metrics["matched_monosaccharides"],
                "motifs": glycan_metrics["found_motifs"],
                "sasa": glycan_metrics["final_sasa"],
                "flexibility": glycan_metrics["final_flexibility"],
                "has_multi_node_motifs": glycan_metrics["has_multi_node_motifs"]  # NEW: Add flag to metrics
            })

            # NEW: If this glycan has multi-node motifs, add it to the testing collection
            if agg_test_data["multi_node_motifs"]:
                agg_test_glycans[glycan] = {
                    "binding_score": binding_score,
                    "motif_data": agg_test_data["motif_parts_data"]
                }

    print(f"Processed {len(metric_data)} glycans with metrics.")
    print(f"Found {len(agg_test_glycans)} glycans with multi-node motifs for aggregation testing.")

    return metric_data, agg_test_glycans


def metric_df(lectin, properties):
    """
    Generate a metrics DataFrame for a given lectin and its properties.
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
        """Filter the binding DataFrame for the given lectin."""
        filtered_df = binding_df[binding_df.iloc[:, -1].eq(lectin)]
        filtered_df = filtered_df.dropna(axis=1, how='all')  # Drop columns with all NaN values
        return filtered_df

    def get_glycan_scores(filtered_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate mean binding scores for glycans."""
        lectin_df = filtered_df.drop(columns=["target", "protein"])  # Exclude "protein" and "target" columns
        glycan_scores = lectin_df.mean(axis=0).to_dict()
        return glycan_scores

    flex_data, binding_df, invalid_graphs = load_data()
    filtered_df = filter_binding_data(binding_df, lectin)
    if filtered_df.empty:
        print(f"No binding data found for {lectin}.")
        return pd.DataFrame(), {}

    glycan_scores = get_glycan_scores(filtered_df)

    # NEW: Get both metrics and aggregation test data
    metric_data, agg_test_glycans = generate_metrics_for_glycan(properties, glycan_scores, flex_data)

    metric_df = pd.DataFrame(metric_data)
    if 'glycan' in metric_df.columns:
        metric_df.set_index('glycan', inplace=True)
    else:
        print(f"⚠️ Warning: 'glycan' column missing for {lectin}. Skipping index setting.")

    # Save to Excel
    metric_df.to_excel(f'results/metric_df/{lectin}_metrics.xlsx', index=True, header=True)

    # NEW: Save aggregation test data to a separate file
    if agg_test_glycans:
        # Create a simple summary DataFrame with just the key information
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


def analyze_aggregation_methods(all_agg_test_glycans):
    """
    Analyzes different combinations of aggregation methods at two levels:
    1. Within motif (combining multiple nodes in a motif part)
    2. Across motifs (combining different motifs in a glycan)

    Returns correlation results for each combination.
    """
    # Results storage
    results = []

    # Aggregation methods to test at both levels
    agg_methods = ['sum', 'mean', 'max']

    # Loop through each lectin
    for lectin, glycans in all_agg_test_glycans.items():
        print(f"Analyzing {lectin} with {len(glycans)} glycans")

        # For each combination of aggregation methods
        for within_method in agg_methods:
            for across_method in agg_methods:
                # Store correlation data for this combination
                combo_results = {
                    'lectin': lectin,
                    'within_method': within_method,
                    'across_method': across_method,
                    'sasa_pearson_r': None, 'sasa_pearson_p': None,
                    'sasa_spearman_r': None, 'sasa_spearman_p': None,
                    'flex_pearson_r': None, 'flex_pearson_p': None,
                    'flex_spearman_r': None, 'flex_spearman_p': None
                }

                # Lists to store binding scores and aggregated property values
                binding_scores = []
                aggregated_sasa = []
                aggregated_flex = []

                # Process each glycan
                for glycan, data in glycans.items():
                    binding_score = data['binding_score']

                    # Lists to store aggregated values for each motif
                    motif_sasa_values = []
                    motif_flex_values = []

                    # Process each motif in this glycan
                    for motif in data['motif_data']:
                        # Lists to store values for parts of this motif
                        part_sasa_values = []
                        part_flex_values = []

                        # Process each part of this motif
                        for part in motif:
                            # Raw SASA and flexibility values for this part
                            sasa_values = part.get('SASA', [])
                            flex_values = part.get('flexibility', [])

                            if not sasa_values or not flex_values:
                                continue

                            # First level aggregation (within motif part)
                            if within_method == 'sum':
                                part_sasa = sum(sasa_values)
                                part_flex = sum(flex_values)
                            elif within_method == 'mean':
                                part_sasa = sum(sasa_values) / len(sasa_values)
                                part_flex = sum(flex_values) / len(flex_values)
                            elif within_method == 'max':
                                part_sasa = max(sasa_values)
                                part_flex = max(flex_values)

                            part_sasa_values.append(part_sasa)
                            part_flex_values.append(part_flex)

                        # Skip if no parts had values
                        if not part_sasa_values or not part_flex_values:
                            continue

                        # Second level aggregation (across motif parts)
                        if within_method == 'sum':
                            motif_sasa = sum(part_sasa_values)
                            motif_flex = sum(part_flex_values)
                        elif within_method == 'mean':
                            motif_sasa = sum(part_sasa_values) / len(part_sasa_values)
                            motif_flex = sum(part_flex_values) / len(part_flex_values)
                        elif within_method == 'max':
                            motif_sasa = max(part_sasa_values)
                            motif_flex = max(part_flex_values)

                        motif_sasa_values.append(motif_sasa)
                        motif_flex_values.append(motif_flex)

                    # Skip if no motifs had values
                    if not motif_sasa_values or not motif_flex_values:
                        continue

                    # Final aggregation (across motifs)
                    if across_method == 'sum':
                        final_sasa = sum(motif_sasa_values)
                        final_flex = sum(motif_flex_values)
                    elif across_method == 'mean':
                        final_sasa = sum(motif_sasa_values) / len(motif_sasa_values)
                        final_flex = sum(motif_flex_values) / len(motif_flex_values)
                    elif across_method == 'max':
                        final_sasa = max(motif_sasa_values)
                        final_flex = max(motif_flex_values)

                    # Store values for correlation analysis
                    binding_scores.append(binding_score)
                    aggregated_sasa.append(final_sasa)
                    aggregated_flex.append(final_flex)

                # Calculate correlations if we have enough data points
                if len(binding_scores) >= 5:
                    # Pearson correlation for SASA
                    sasa_pearson = pearsonr(aggregated_sasa, binding_scores)
                    combo_results['sasa_pearson_r'] = sasa_pearson[0]
                    combo_results['sasa_pearson_p'] = sasa_pearson[1]

                    # Spearman correlation for SASA
                    sasa_spearman = spearmanr(aggregated_sasa, binding_scores)
                    combo_results['sasa_spearman_r'] = sasa_spearman[0]
                    combo_results['sasa_spearman_p'] = sasa_spearman[1]

                    # Pearson correlation for flexibility
                    flex_pearson = pearsonr(aggregated_flex, binding_scores)
                    combo_results['flex_pearson_r'] = flex_pearson[0]
                    combo_results['flex_pearson_p'] = flex_pearson[1]

                    # Spearman correlation for flexibility
                    flex_spearman = spearmanr(aggregated_flex, binding_scores)
                    combo_results['flex_spearman_r'] = flex_spearman[0]
                    combo_results['flex_spearman_p'] = flex_spearman[1]

                    # Print results for this combination
                    print(f"  {within_method} within / {across_method} across:")
                    print(
                        f"    SASA: Pearson r={sasa_pearson[0]:.4f} (p={sasa_pearson[1]:.4f}), Spearman r={sasa_spearman[0]:.4f} (p={sasa_spearman[1]:.4f})")
                    print(
                        f"    FLEX: Pearson r={flex_pearson[0]:.4f} (p={flex_pearson[1]:.4f}), Spearman r={flex_spearman[0]:.4f} (p={flex_spearman[1]:.4f})")

                # Add results for this combination
                results.append(combo_results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Add significance indicators
    results_df['sasa_pearson_sig'] = results_df['sasa_pearson_p'] < 0.05
    results_df['sasa_spearman_sig'] = results_df['sasa_spearman_p'] < 0.05
    results_df['flex_pearson_sig'] = results_df['flex_pearson_p'] < 0.05
    results_df['flex_spearman_sig'] = results_df['flex_spearman_p'] < 0.05

    return results_df


def find_optimal_aggregation_methods(results_df):
    """
    Identifies the optimal combination of aggregation methods
    based on correlation strength and significance.
    """
    # Create separate views for each property and correlation type
    property_corr_types = [
        ('sasa', 'pearson'), ('sasa', 'spearman'),
        ('flex', 'pearson'), ('flex', 'spearman')
    ]

    best_methods = {}

    for prop, corr_type in property_corr_types:
        # Filter for significant results only
        col_r = f'{prop}_{corr_type}_r'
        col_sig = f'{prop}_{corr_type}_sig'

        # Get only significant correlations
        sig_results = results_df[results_df[col_sig] == True].copy()

        # If no significant correlations, use all results
        if sig_results.empty:
            sig_results = results_df.copy()
            print(f"Warning: No significant {corr_type} correlations for {prop}")

        # Find the strongest absolute correlation
        sig_results['abs_corr'] = sig_results[col_r].abs()
        best_idx = sig_results['abs_corr'].idxmax()
        best_row = sig_results.loc[best_idx]

        best_methods[f'{prop}_{corr_type}'] = {
            'within_method': best_row['within_method'],
            'across_method': best_row['across_method'],
            'correlation': best_row[col_r],
            'p_value': best_row[f'{prop}_{corr_type}_p'],
            'significant': best_row[col_sig]
        }

    # Determine overall best method
    method_votes = {}
    for key, method_info in best_methods.items():
        combo = (method_info['within_method'], method_info['across_method'])
        method_votes[combo] = method_votes.get(combo, 0) + 1

    best_combo = max(method_votes.items(), key=lambda x: x[1])[0]

    print(f"Overall best aggregation combination: {best_combo[0]} within / {best_combo[1]} across")
    print("Best methods by property and correlation type:")
    for key, method_info in best_methods.items():
        print(
            f"  {key}: {method_info['within_method']} within / {method_info['across_method']} across (r={method_info['correlation']:.4f}, p={method_info['p_value']:.4f})")

    return best_methods, best_combo

# First, ensure the directory exists
def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# 1. Aggregation Heatmap
def create_aggregation_heatmap(results_df):
    """Creates a heatmap visualization of aggregation method performance"""
    # Ensure directory exists
    ensure_dir("results/plots")

    # Prepare data for heatmap
    within_methods = ['sum', 'mean', 'max']
    across_methods = ['sum', 'mean', 'max']

    # Create separate heatmaps for each property and correlation type
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    metrics = [
        ('sasa_pearson_r', 'SASA - Pearson', axes[0, 0]),
        ('sasa_spearman_r', 'SASA - Spearman', axes[0, 1]),
        ('flex_pearson_r', 'Flexibility - Pearson', axes[1, 0]),
        ('flex_spearman_r', 'Flexibility - Spearman', axes[1, 1])
    ]

    for metric, title, ax in metrics:
        # Create a heatmap matrix
        heatmap_data = np.zeros((3, 3))
        significance_mask = np.zeros((3, 3), dtype=bool)

        # Fill the matrix with correlation values
        for i, within in enumerate(within_methods):
            for j, across in enumerate(across_methods):
                # Filter for this combination
                combo_data = results_df[
                    (results_df['within_method'] == within) &
                    (results_df['across_method'] == across)
                    ]

                if not combo_data.empty:
                    # Calculate average correlation across all lectins for this combination
                    avg_corr = combo_data[metric].mean()
                    heatmap_data[i, j] = avg_corr

                    # Mark significant correlations
                    p_value_col = metric.replace('_r', '_p')
                    avg_sig = (combo_data[p_value_col] < 0.05).sum() / len(combo_data)
                    significance_mask[i, j] = avg_sig > 0.5  # Mark if majority are significant

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="coolwarm" if "flex" not in metric else "coolwarm_r",
            center=0,
            vmin=-1, vmax=1,
            xticklabels=across_methods,
            yticklabels=within_methods,
            ax=ax
        )

        # Highlight significant combinations
        for i in range(3):
            for j in range(3):
                if significance_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

        ax.set_title(title)
        ax.set_xlabel('Across Motifs Method')
        ax.set_ylabel('Within Motif Method')

    plt.tight_layout()
    plt.savefig('results/plots/aggregation_method_comparison_heatmap.png', dpi=300)
    print("Saved heatmap to results/plots/aggregation_method_comparison_heatmap.png")
    plt.show()


# 2. Scatter Plot Matrix for Individual Lectins
def compare_lectin_aggregation_scatterplots(all_agg_test_glycans, lectin_name):
    """Creates scatter plots comparing aggregation methods for a specific lectin"""
    # Ensure directory exists
    ensure_dir("results/plots")

    glycans = all_agg_test_glycans[lectin_name]
    binding_scores = [data['binding_score'] for glycan, data in glycans.items()]

    # Methods to compare
    within_methods = ['sum', 'mean', 'max']
    across_methods = ['sum', 'mean', 'max']
    combinations = [(w, a) for w in within_methods for a in across_methods]

    # For each property (SASA and flexibility)
    properties = ['SASA', 'flexibility']

    for property_name in properties:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)

        # Set global min/max for consistent y-axis scaling
        all_values = []
        for within_method, across_method in combinations:
            property_values = calculate_aggregated_values(
                glycans, property_name, within_method, across_method
            )
            all_values.extend(property_values)

        y_min, y_max = min(all_values), max(all_values)

        # Create scatter plots for each combination
        for i, within_method in enumerate(within_methods):
            for j, across_method in enumerate(across_methods):
                ax = axes[i, j]

                # Calculate aggregated property values
                property_values = calculate_aggregated_values(
                    glycans, property_name, within_method, across_method
                )

                # Create scatter plot
                ax.scatter(binding_scores, property_values, alpha=0.7)

                # Add trend line
                if len(binding_scores) >= 2:
                    r, p = pearsonr(binding_scores, property_values)
                    slope, intercept, _, _, _ = linregress(binding_scores, property_values)
                    x_range = np.linspace(min(binding_scores), max(binding_scores), 100)
                    ax.plot(x_range, slope * x_range + intercept, 'r--')

                    # Add correlation coefficient and p-value
                    ax.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.3f}",
                            transform=ax.transAxes, va='top', ha='left')

                # Set axis labels for edge plots
                if i == 2:
                    ax.set_xlabel('Binding Score')
                if j == 0:
                    ax.set_ylabel(f'{property_name} Value')

                # Set title
                ax.set_title(f'{within_method} within / {across_method} across')

                # Set consistent y-axis limits
                ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))

        plt.tight_layout()
        plt.suptitle(f'{lectin_name}: {property_name} vs Binding Score', y=1.02, fontsize=16)

        # Save to results/plots directory
        plt.savefig(f'results/plots/{lectin_name}_{property_name}_aggregation_comparison.png', dpi=300)
        print(f"Saved scatter plot to results/plots/{lectin_name}_{property_name}_aggregation_comparison.png")
        plt.show()


# Helper function for scatter plots
def calculate_aggregated_values(glycans, property_name, within_method, across_method):
    """Helper function to calculate aggregated values with a given method combination"""
    property_values = []

    for glycan, data in glycans.items():
        # Lists to store per-motif aggregated values
        motif_values = []

        # Process each motif in this glycan
        for motif in data['motif_data']:
            # Skip empty motifs
            if not motif:
                continue

            # Lists for part values within this motif
            part_values = []

            # Process each part of this motif
            for part in motif:
                # Extract raw values
                raw_values = part.get(property_name, [])

                # Skip if no values
                if not raw_values:
                    continue

                # First level aggregation (within part)
                if within_method == 'sum':
                    part_value = sum(raw_values)
                elif within_method == 'mean':
                    part_value = sum(raw_values) / len(raw_values)
                else:  # max
                    part_value = max(raw_values) if raw_values else 0

                # Add to part values for this motif
                part_values.append(part_value)

            # Skip if no parts had values
            if not part_values:
                continue

            # Second level aggregation (across parts within motif)
            if across_method == 'sum':
                motif_value = sum(part_values)
            elif across_method == 'mean':
                motif_value = sum(part_values) / len(part_values)
            else:  # max
                motif_value = max(part_values)

            # Add to motif values for this glycan
            motif_values.append(motif_value)

        # Skip glycans with no valid motifs
        if not motif_values:
            continue

        # Final aggregation is always max across motifs (consistent with original code)
        final_value = max(motif_values)
        property_values.append(final_value)

    return property_values


# 3. Parallel Coordinates Plot
def create_parallel_coordinates(results_df):
    """Creates a parallel coordinates plot to visualize relationships between aggregation methods"""
    # Ensure directory exists
    ensure_dir("results/plots")

    try:
        import plotly.express as px
        import plotly.io as pio

        # Prepare data
        plot_data = []

        # Aggregation combinations
        within_methods = ['sum', 'mean', 'max']
        across_methods = ['sum', 'mean', 'max']

        for within in within_methods:
            for across in across_methods:
                # Filter for this combination
                combo_data = results_df[
                    (results_df['within_method'] == within) &
                    (results_df['across_method'] == across)
                    ]

                if not combo_data.empty:
                    # Calculate averages across all lectins
                    avg_row = {
                        'within_method': within,
                        'across_method': across,
                        'combo': f"{within}-{across}",
                        'sasa_pearson_r': combo_data['sasa_pearson_r'].mean(),
                        'sasa_spearman_r': combo_data['sasa_spearman_r'].mean(),
                        'flex_pearson_r': combo_data['flex_pearson_r'].mean(),
                        'flex_spearman_r': combo_data['flex_spearman_r'].mean(),
                        'sasa_pearson_sig_pct': (combo_data['sasa_pearson_p'] < 0.05).mean() * 100,
                        'flex_pearson_sig_pct': (combo_data['flex_pearson_p'] < 0.05).mean() * 100
                    }
                    plot_data.append(avg_row)

        # Convert to DataFrame
        plot_df = pd.DataFrame(plot_data)

        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            plot_df,
            dimensions=['sasa_pearson_r', 'sasa_spearman_r', 'flex_pearson_r',
                        'flex_spearman_r', 'sasa_pearson_sig_pct', 'flex_pearson_sig_pct'],
            color='combo',
            labels={
                'sasa_pearson_r': 'SASA Pearson',
                'sasa_spearman_r': 'SASA Spearman',
                'flex_pearson_r': 'Flex Pearson',
                'flex_spearman_r': 'Flex Spearman',
                'sasa_pearson_sig_pct': '% SASA Significant',
                'flex_pearson_sig_pct': '% Flex Significant',
            },
            title='Comparison of Aggregation Method Combinations Across Multiple Metrics'
        )

        # Save to results/plots directory
        pio.write_html(fig, 'results/plots/aggregation_parallel_coords.html')
        print("Saved parallel coordinates plot to results/plots/aggregation_parallel_coords.html")
        return fig
    except ImportError:
        print("Could not create parallel coordinates plot - plotly is required. Install with: pip install plotly")
        return None


# 4. Hierarchical Clustering of Methods
def cluster_aggregation_methods(results_df):
    """Clusters aggregation methods based on similarity of their correlation results"""
    # Ensure directory exists
    ensure_dir("results/plots")

    # Prepare data for clustering
    methods_data = []
    method_labels = []

    within_methods = ['sum', 'mean', 'max']
    across_methods = ['sum', 'mean', 'max']
    metrics = ['sasa_pearson_r', 'sasa_spearman_r', 'flex_pearson_r', 'flex_spearman_r']

    for within in within_methods:
        for across in across_methods:
            combo_data = results_df[
                (results_df['within_method'] == within) &
                (results_df['across_method'] == across)
                ]

            if not combo_data.empty:
                # Create feature vector of correlation values for this method
                features = []
                for metric in metrics:
                    # Calculate average correlation across all lectins
                    features.append(combo_data[metric].mean())

                methods_data.append(features)
                method_labels.append(f"{within}-{across}")

    # Convert to numpy array
    methods_array = np.array(methods_data)

    # Perform hierarchical clustering
    Z = linkage(methods_array, 'ward')

    # Create dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=method_labels, leaf_rotation=90)
    plt.title('Hierarchical Clustering of Aggregation Methods')
    plt.xlabel('Aggregation Method Combination')
    plt.ylabel('Distance')
    plt.tight_layout()

    # Save to results/plots directory
    plt.savefig('results/plots/aggregation_method_clustering.png', dpi=300)
    print("Saved clustering dendrogram to results/plots/aggregation_method_clustering.png")
    plt.show()


# 5. Side-by-Side Property Comparisons
def compare_property_effects(results_df):
    """Compare how different aggregation methods affect SASA vs. flexibility correlations"""
    # Ensure directory exists
    ensure_dir("results/plots")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get average correlation values for each method combination
    within_methods = ['sum', 'mean', 'max']
    across_methods = ['sum', 'mean', 'max']

    # Prepare data
    combinations = []
    sasa_pearson_values = []
    flex_pearson_values = []
    significance = []

    for within in within_methods:
        for across in across_methods:
            combo_data = results_df[
                (results_df['within_method'] == within) &
                (results_df['across_method'] == across)
                ]

            if not combo_data.empty:
                combinations.append(f"{within}\n{across}")

                # Get average correlations
                sasa_pearson_values.append(combo_data['sasa_pearson_r'].mean())
                flex_pearson_values.append(combo_data['flex_pearson_r'].mean())

                # Calculate significance percentage
                sasa_sig = (combo_data['sasa_pearson_p'] < 0.05).mean()
                flex_sig = (combo_data['flex_pearson_p'] < 0.05).mean()
                significance.append((sasa_sig + flex_sig) / 2)  # Average significance

    # Create bar plots
    x = np.arange(len(combinations))
    width = 0.35

    # SASA Pearson correlations
    bars1 = ax1.bar(x, sasa_pearson_values, width, alpha=0.7, label='SASA')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_title('SASA Pearson Correlation by Aggregation Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(combinations, rotation=45, ha='right')

    # Highlight the best method for SASA
    best_idx = np.argmax(np.abs(sasa_pearson_values))
    bars1[best_idx].set_color('darkgreen')
    bars1[best_idx].set_hatch('///')

    # Flexibility Pearson correlations
    bars2 = ax2.bar(x, flex_pearson_values, width, alpha=0.7, label='Flexibility')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Flexibility Pearson Correlation by Aggregation Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(combinations, rotation=45, ha='right')

    # Highlight the best method for flexibility
    best_idx = np.argmax(np.abs(flex_pearson_values))
    bars2[best_idx].set_color('darkred')
    bars2[best_idx].set_hatch('///')

    # Add significance markers
    for i, (bar1, bar2, sig) in enumerate(zip(bars1, bars2, significance)):
        if sig > 0.5:  # If more than 50% of correlations are significant
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax1.text(bar1.get_x() + bar1.get_width() / 2., height1,
                     '*', ha='center', va='bottom', fontsize=14)
            ax2.text(bar2.get_x() + bar2.get_width() / 2., height2,
                     '*', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()

    # Save to results/plots directory
    plt.savefig('results/plots/sasa_vs_flex_method_comparison.png', dpi=300)
    print("Saved property comparison to results/plots/sasa_vs_flex_method_comparison.png")
    plt.show()


# Main function to generate all plots
def generate_all_plots(results_df, all_agg_test_glycans):
    """Generate all visualization plots and save them to results/plots directory"""
    # First ensure the directory exists
    ensure_dir("results/plots")
    print("Generating all visualization plots...")

    # 1. Create aggregation heatmap
    create_aggregation_heatmap(results_df)

    # 2. Create property comparisons
    compare_property_effects(results_df)

    # 3. Create hierarchical clustering
    cluster_aggregation_methods(results_df)

    # 4. Create parallel coordinates plot (if plotly is available)
    create_parallel_coordinates(results_df)

    # 5. Create scatter plots for top lectins (choose lectins with sufficient data)
    lectins_with_data = []
    for lectin, glycans in all_agg_test_glycans.items():
        if len(glycans) >= 10:  # Only include lectins with at least 10 glycans
            lectins_with_data.append((lectin, len(glycans)))

    # Sort by number of glycans and select top 3
    lectins_with_data.sort(key=lambda x: x[1], reverse=True)
    for lectin, count in lectins_with_data[:3]:
        print(f"Creating scatter plots for {lectin} ({count} glycans)...")
        compare_lectin_aggregation_scatterplots(all_agg_test_glycans, lectin)

    print("All plots have been saved to the results/plots directory.")