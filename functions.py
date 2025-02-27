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

BINDING_DATA_PATH_v3 = 'data/20250216_glycan_binding.csv'
BINDING_DATA_PATH_v2 = 'data/20241206_glycan_binding.csv'  # seq,protein
BINDING_DATA_PATH_v1 = 'data/glycan_binding.csv'  # seq,protein


def load_data_pdb():
    """Load glycan flexibility data from PDB source."""
    with pkg_resources.files(glycontact).joinpath("glycan_graphs.pkl").open("rb") as f:
        return pickle.load(f)


def load_data():
    """Load glycan flexibility and binding data, process graphs, and return results."""
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

        # ƒApply mapping to create the "protein" column in df_target
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


def get_glycan_scores(filtered_df: dict[str, float]) -> Dict[str, float]:
    """Calculate mean binding scores for glycans."""
    lectin_df = filtered_df.drop(columns=["target", "protein"])  # Exclude "protein" and "target" columns
    glycan_scores = lectin_df.mean(axis=0).to_dict()

    return glycan_scores


def find_matching_glycan(flex_data, glycan):
    """Find the matching glycan in flex_data."""
    for flex_glycan in flex_data.keys():
        if compare_glycans(glycan, flex_glycan):
            return glycan
    return None


def compute_overall_SASA(SASA_values):
    return sum(SASA_values) / len(SASA_values) if SASA_values else None


def compute_SASA_stats(SASA_values):
    if not SASA_values:  # Check if list is empty
        return None, None, None

    sasa_sum = sum(SASA_values)
    sasa_mean = sasa_sum / len(SASA_values)
    sasa_max = max(SASA_values)

    return {
        "sasa_mean": sasa_mean,
        "sasa_max": sasa_max,
        "sasa_sum": sasa_sum
    }


def compute_overall_flexibility(flexibility_values):
    return sum(flexibility_values) / len(flexibility_values) if flexibility_values else None


def compute_overall_Q(Q_values):
    return sum(Q_values) / len(Q_values) if Q_values else None


def compute_overall_theta(theta_values):
    return sum(theta_values) / len(theta_values) if theta_values else None


def process_glycan_with_motifs(matched_glycan: str,
                               properties: dict,
                               flex_data: dict[str, nx.Graph]):
    """
    Process a glycan string to find nodes matching binding motifs and calculate metrics.
    Handles both single and multiple binding motifs.

    Args:
        matched_glycan (str): Identifier of the glycan to process.
        properties (dict): Properties including motifs and termini lists.
        flex_data (dict): Dictionary mapping glycan identifiers to graphs.

    Returns:
        tuple: Matching monosaccharides, SASA-weighted scores, flexibility-weighted scores, and found motifs.
    """
    matching_monosaccharides, SASA, flexibility, Q, theta, conformation, found_motifs = [], [], [], [], [], [], []

    motifs = properties.get("motif", [])
    termini_list = properties.get("termini_list", [])

    for motif, termini in zip(motifs, termini_list):
        try:
            is_present, matched_nodes = subgraph_isomorphism(
                matched_glycan, motif,
                return_matches=True,
                termini_list=termini
            )
            if not is_present:
                continue
        except Exception as e:
            print(f"Subgraph isomorphism error for glycan {matched_glycan} with motif {motif}: {e}")
            continue

        found_motifs.append(motif)
        print(f"Processing motif: {motif} for glycan: {matched_glycan}")

        matched_nodes = [node for sublist in matched_nodes for node in sublist] if matched_nodes and isinstance(
            matched_nodes[0], list) else matched_nodes
        print(f"Matched nodes: {matched_nodes}")

        pdb_graph = flex_data.get(matched_glycan)
        if not isinstance(pdb_graph, nx.Graph):
            print(f"No valid graph found for glycan: {matched_glycan}")
            continue

        # Select only monosaccharides (even-indexed nodes)
        selected_mono = [node for node in matched_nodes if node in pdb_graph.nodes and node % 2 == 0]
        print(f"Selected monosaccharides: {selected_mono}")

        if hasattr(pdb_graph, "nodes"):
            print(f"Graph nodes: {pdb_graph.nodes(data=True)}")
            for mono in selected_mono:
                try:
                    attributes = pdb_graph.nodes[mono]
                    monosaccharide = attributes.get('Monosaccharide', "")
                    if monosaccharide:  # Ensure non-empty monosaccharides
                        matching_monosaccharides.append(monosaccharide)
                        SASA.append(attributes.get("SASA", 0))
                        flexibility.append(attributes.get("flexibility", 0))
                        Q.append(attributes.get("Q", 0))
                        theta.append(attributes.get("theta", 0))
                        conformation.append(attributes.get("conformation", 0))

                    print(f"Matching monosaccharides: {matching_monosaccharides}")
                    print(f"SASA scores: {SASA}")
                    print(f"Flexibility: {flexibility}")
                    print(f"Q: {Q}")
                    print(f"theta: {theta}")
                    print(f"conformation: {conformation}")

                    print("")
                except Exception as e:
                    print(f"Error extracting attributes for node {mono} in glycan {matched_glycan}: {e}")
        else:
            print(f"Skipping invalid graph or graph with no nodes for glycan: {matched_glycan}")

    return matching_monosaccharides, SASA, flexibility, Q, theta, conformation, found_motifs


def generate_metrics_for_glycan(properties: str,
                                glycan_scores: dict,
                                flex_data: dict[str, Graph]) -> list[dict]:
    """
    Generate metrics for each glycan by processing them one by one and creating metrics at the end.
    """
    metric_data = []
    missing_glycans = []

    for glycan in glycan_scores:
        binding_score = glycan_scores[glycan]
        matched_glycan = find_matching_glycan(flex_data, glycan)

        if not matched_glycan:
            missing_glycans.append(glycan)
            continue

        # Process the matched glycan
        matching_monosaccharides, SASA, flexibility, Q, theta, conformation, found_motifs = process_glycan_with_motifs(
            matched_glycan, properties, flex_data)

        # Skip empty monosaccharides
        matching_monosaccharides = [m for m in matching_monosaccharides if m.strip()]

        if matching_monosaccharides:
            overall_SASA = compute_overall_SASA(SASA)


            overall_flexibility = compute_overall_flexibility(flexibility)
            overall_Q = compute_overall_Q(Q)
            overall_theta = compute_overall_theta(theta)
            glycan_class = get_class(matched_glycan) or np.nan

            metric_data.append({
                "glycan": glycan,
                "binding_score": binding_score,
                "SASA": overall_SASA, #,sasa_mean
                "sum_SASA" :compute_SASA_stats(SASA)["sasa_sum"],
                "max_SASA"  :compute_SASA_stats(SASA)["sasa_max"],
                "flexibility": overall_flexibility,
                "Q": overall_Q,
                "theta": overall_theta,
                "conformation": conformation,
                "monosaccharides": matching_monosaccharides,
                "motifs": found_motifs,
                "class": glycan_class,
            })

    print(f"Processed {len(metric_data)} glycans with metrics.")

    # Report missing glycans
    if missing_glycans:
        print(f"Not-matched glycan in flex data: {missing_glycans}")

    return metric_data


def metric_df(lectin, properties):
    """
    Generate a metrics DataFrame for a given lectin and its properties.
    """
    flex_data, binding_df, invalid_graphs = load_data()
    filtered_df = filter_binding_data(binding_df, lectin)
    if filtered_df.empty:
        print(f"No binding data found for {lectin}.")
        return pd.DataFrame()

    glycan_scores = get_glycan_scores(filtered_df)
    metric_data = generate_metrics_for_glycan(properties, glycan_scores, flex_data)

    metric_df = pd.DataFrame(metric_data)
    if 'glycan' in metric_df.columns:
        metric_df.set_index('glycan', inplace=True)
    else:
        print(f"⚠️ Warning: 'glycan' column missing for {lectin}. Skipping index setting.")

    metric_df.to_excel(f'results/metric_df/{lectin}_metrics.xlsx', index=True, header=True)
    return metric_df


sys.stdout = open('results/metric_df/metric_df.log', 'w')

def plot_Binding_vs_Flexibility_and_SASA_with_stats(metric_df, lectin, binding_motif):
    """Plots Binding vs Flexibility and Binding vs SASA with colors for glycans,
    and annotates the plots with formal regression coefficients and associated p-values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Create two subplots side by side

    # -------------------------------
    # Plot 1: Binding vs Flexibility
    # -------------------------------
    # Scatter plot colored by glycan class
    sns.scatterplot(
        ax=axes[0],
        x='flexibility',
        y='binding_score',
        data=metric_df,
        hue="class",
        hue_order=['N', 'O', "free", "lipid", "lipid/free", ""],
        palette="tab10",
        alpha=0.7
    )
    # Regression line using seaborn
    sns.regplot(
        ax=axes[0],
        x='flexibility',
        y='binding_score',
        data=metric_df,
        scatter=False,  # Do not plot points again
        line_kws={'color': 'red'}
    )
    axes[0].set_title(f'Binding vs Flexibility\n{lectin} {binding_motif}', fontsize=12)
    axes[0].set_xlabel('Flexibility')
    axes[0].set_ylabel('Binding Score')
    axes[0].get_legend().remove()  # Remove legend from the first plot

    # Compute regression parameters for Binding vs Flexibility
    flex = metric_df['flexibility']
    binding = metric_df['binding_score']
    slope_flex, intercept_flex, r_value_flex, p_value_flex, std_err_flex = linregress(flex, binding)

    # Create a text string for the annotation
    text_flex = (f'y = {intercept_flex:.2f} + {slope_flex:.2f}x\n'
                 f'p-value = {p_value_flex:.3g}')
    # Annotate the first subplot with the regression info
    axes[0].text(0.05, 0.95, text_flex, transform=axes[0].transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

    # -------------------------
    # Plot 2: Binding vs SASA
    # -------------------------
    # Scatter plot colored by glycan class
    sns.scatterplot(
        ax=axes[1],
        x='SASA',
        y='binding_score',
        data=metric_df,
        hue='class',
        hue_order=['N', 'O', "free", "lipid", "lipid/free", ""],
        palette="tab10",
        alpha=0.7
    )
    # Regression line using seaborn
    sns.regplot(
        ax=axes[1],
        x='SASA',
        y='binding_score',
        data=metric_df,
        scatter=False,  # Do not plot points again
        line_kws={'color': 'red'}
    )
    axes[1].set_title(f'Binding vs SASA\n{lectin} {binding_motif}', fontsize=12)
    axes[1].set_xlabel('SASA')
    axes[1].set_ylabel('Binding Score')
    axes[1].legend(title="Glycan class", bbox_to_anchor=(1.05, 1), loc='upper left')  # Keep class legend

    # Compute regression parameters for Binding vs SASA
    sasa = metric_df['SASA']
    slope_sasa, intercept_sasa, r_value_sasa, p_value_sasa, std_err_sasa = linregress(sasa, binding)

    # Create a text string for the annotation
    text_sasa = (f'y = {intercept_sasa:.2f} + {slope_sasa:.2f}x\n'
                 f'p-value = {p_value_sasa:.3g}')
    # Annotate the second subplot with the regression info
    axes[1].text(0.05, 0.95, text_sasa, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'results/plots/Binding_vs_Flexibility_and_SASA_{lectin}.png', dpi=300)

    # Show the plots
    plt.show()


def analyze_all_lectins(metric_df_dict):
    """
    Analyzes the correlation between Binding Score and SASA/Flexibility for all lectins,
    groups them by correlation status (Positive, Negative, Not Significant),
    and saves results to a single Excel file.

    Args:
        metric_df_dict (dict): Dictionary where keys are lectin names and values are metric DataFrames.

    Returns:
        pd.DataFrame: A DataFrame containing all lectins with their p-values and correlation status.
    """
    results = []

    for lectin, metric_df in metric_df_dict.items():

        # Compute regression for SASA vs Binding Score
        sasa = metric_df["SASA"]
        #sasa = metric_df["sum_SASA"]
        #sasa = metric_df["max_SASA"]

        binding = metric_df["binding_score"]
        slope_sasa, _, _, p_value_sasa, _ = linregress(sasa, binding)

        # Compute regression for Flexibility vs Binding Score
        flex = metric_df["flexibility"]
        slope_flex, _, _, p_value_flex, _ = linregress(flex, binding)

        # Determine correlation status for SASA
        correlation_status_sasa = (
            "Positive Correlation" if p_value_sasa < 0.05 and slope_sasa > 0 else
            "Negative Correlation" if p_value_sasa < 0.05 and slope_sasa < 0 else
            "Not Significant"
        )

        # Determine correlation status for Flexibility
        correlation_status_flex = (
            "Positive Correlation" if p_value_flex < 0.05 and slope_flex > 0 else
            "Negative Correlation" if p_value_flex < 0.05 and slope_flex < 0 else
            "Not Significant"
        )

        # Append results for SASA and Flexibility (only if significant)
        if correlation_status_sasa != "Not Significant":
            results.append({"Lectin": lectin, "p-value": p_value_sasa, "Correlation Status": correlation_status_sasa,
                            "Type": "SASA"})

        if correlation_status_flex != "Not Significant":
            results.append({"Lectin": lectin, "p-value": p_value_flex, "Correlation Status": correlation_status_flex,
                            "Type": "Flexibility"})

        # Create DataFrame
    results_df = pd.DataFrame(results)

    # Rank lectins within each correlation group by p-value (ascending order)
    results_df["Rank"] = results_df.groupby("Correlation Status")["p-value"].rank(method="min", ascending=True)

    results_df = results_df.sort_values(by="Rank", ascending=True)

    # Set "Correlation Status" as the index to group in one file
    results_df.set_index("Correlation Status", inplace=True)

    # Save to a single Excel file
    excel_filename = "results/stats/all_lectin_correlation_mean_SASA.xlsx"
    results_df.to_excel(excel_filename)

    return results_df
