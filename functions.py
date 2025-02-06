import pickle
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import pingouin as pg
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.graph import  glycan_to_nxGraph
from glycowork.motif.processing import get_class
import sys
from networkx.classes import Graph
import networkx as nx
from sklearn.preprocessing import LabelEncoder

FLEX_DATA_PATH = 'data/glycan_graphs.pkl'
BINDING_DATA_PATH = 'data/20241206_glycan_binding.csv'


def load_data_pdb():
    """Load glycan flexibility data from PDB source."""
    with open(FLEX_DATA_PATH, 'rb') as file:
        return pickle.load(file)

def remove_and_concatenate_labels(graph):
    """Modify the graph by removing and concatenating labels for odd-indexed nodes."""
    nodes_to_remove = []
    for node in sorted(graph.nodes):
        if node % 2 == 1:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) > 1:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        graph.add_edge(neighbors[i], neighbors[j])
            predecessor = node - 1
            if predecessor in graph.nodes:
                predecessor_label = graph.nodes[predecessor].get("string_labels", "")
                current_label = graph.nodes[node].get("string_labels", "")
                graph.nodes[predecessor]["string_labels"] = f"{predecessor_label}({current_label})"
            nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)

def trim_gcontact(graph):
    """Trim the G_contact graph by removing node 1 and reconnecting its neighbors."""
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected G_contact to be a networkx.Graph, but got {type(graph).__name__}")
    if 1 in graph:
        neighbors = list(graph.neighbors(1))
        if len(neighbors) > 1:
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    graph.add_edge(neighbors[i], neighbors[j])
        graph.remove_node(1)

def compare_graphs_with_attributes(G_contact, G_work):
    """Compare two graphs using node attributes and return a mapping dictionary."""

    def node_match(attrs1, attrs2):
        return 'string_labels' in attrs1 and 'Monosaccharide' in attrs2 and attrs1['string_labels'] in attrs2[
            'Monosaccharide']

    matcher = nx.isomorphism.GraphMatcher(G_work, G_contact, node_match=node_match)
    mapping_dict = {node_g2: node_g for node_g, node_g2 in matcher.mapping.items()} if matcher.is_isomorphic() else {}
    return mapping_dict

def create_glycontact_annotated_graph(glycan: str, mapping_dict, flex_data_pdb_g) -> nx.Graph:
    """Create an annotated glyco-contact graph with flexibility attributes."""
    if glycan not in flex_data_pdb_g or not isinstance(flex_data_pdb_g[glycan], nx.Graph):
        raise ValueError(f"Invalid glycan input or glycan not found in flex_data_pdb_g: {glycan}")

    glycowork_graph = glycan_to_nxGraph(glycan)
    try:
        node_attributes = {node: flex_data_pdb_g[glycan].nodes[node] for node in flex_data_pdb_g[glycan].nodes}
    except KeyError:
        raise KeyError(f'The glycan {glycan} is not present in the flex database')

    mapped_attributes = {mapping_dict[node]: attr for node, attr in node_attributes.items() if node in mapping_dict}
    nx.set_node_attributes(glycowork_graph, mapped_attributes)
    return glycowork_graph

def load_data():
    """Load glycan flexibility and binding data, process graphs, and return results."""
    flex_data = load_data_pdb()
    binding_df = pd.read_csv(BINDING_DATA_PATH)
    invalid_graphs = [glycan for glycan in flex_data if not isinstance(flex_data[glycan], nx.Graph)]
    G_mapped = {}

    for glycan, G_contact in flex_data.items():
        if not hasattr(G_contact, 'neighbors') or not G_contact:
            invalid_graphs.append(glycan)
            continue

        try:
            G_work = glycan_to_nxGraph(glycan)
        except Exception as e:
            print(f"Error converting glycan {glycan} to networkx graph: {e}")
            continue

        remove_and_concatenate_labels(G_work)
        #G_contact = G_work.copy()
        trim_gcontact(G_contact)
        m_dict = compare_graphs_with_attributes(G_contact, G_work)
        G_mapped[glycan] = create_glycontact_annotated_graph(glycan, m_dict, flex_data)

    return G_mapped, binding_df, invalid_graphs

def filter_binding_data(binding_df: pd.DataFrame, lectin: str) -> pd.DataFrame:
    """Filter the binding DataFrame for the given lectin."""
    filtered_df = binding_df[binding_df.iloc[:, -1].eq(lectin)]
    filtered_df = filtered_df.dropna(axis=1, how='all')  # Drop columns with all NaN values
    return filtered_df

def get_glycan_scores(filtered_df: dict[str, float]) -> Dict[str, float]:
    """Calculate mean binding scores for glycans."""
    lectin_df = filtered_df.iloc[:, :-2]  # Exclude "protein" and "target" columns
    glycan_scores = lectin_df.mean(axis=0).to_dict()
    return glycan_scores

def find_matching_glycan(flex_data , glycan):
    """Find the matching glycan in flex_data."""
    for flex_glycan in flex_data.keys():
        if compare_glycans(glycan, flex_glycan):
            return glycan
    return None

def compute_sasa_metrics(nodes):
    """Compute sum, mean, and max SASA metrics for a list of numeric scores."""
    if not nodes:
        return {"SASA_weighted": None, "SASA_weighted_max": None, "SASA_weighted_sum": None}
    SASA_weighted_sum = sum(nodes)
    SASA_weighted = SASA_weighted_sum / len(nodes)
    SASA_weighted_max = max(nodes)
    return {"SASA_weighted": SASA_weighted, "SASA_weighted_max": SASA_weighted_max, "SASA_weighted_sum": SASA_weighted_sum}

def compute_overall_flexibility(flexibility_values ):

    return sum(flexibility_values) / len(flexibility_values) if flexibility_values else None

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
    matching_monosaccharides, sasa_weighted, flexibility_weighted, found_motifs = [], [], [], []

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

        matched_nodes = [node for sublist in matched_nodes for node in sublist] if matched_nodes and isinstance(matched_nodes[0], list) else matched_nodes
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
                        sasa_weighted.append(attributes.get("Weighted Score", 0))
                        flexibility_weighted.append(attributes.get("weighted_mean_flexibility", 0))

                    print(f"Matching monosaccharides: {matching_monosaccharides}")
                    print(f"SASA-weighted scores: {sasa_weighted}")
                    print(f"Flexibility-weighted scores: {flexibility_weighted}")
                    print("")
                except Exception as e:
                    print(f"Error extracting attributes for node {mono} in glycan {matched_glycan}: {e}")
        else:
            print(f"Skipping invalid graph or graph with no nodes for glycan: {matched_glycan}")

    return matching_monosaccharides, sasa_weighted, flexibility_weighted, found_motifs

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
        matching_monosaccharides, sasa_weighted, flexibility_weighted, found_motifs = process_glycan_with_motifs(
            matched_glycan, properties, flex_data)

        # Skip empty monosaccharides
        matching_monosaccharides = [m for m in matching_monosaccharides if m.strip()]

        if matching_monosaccharides:
            sasa_metrics = compute_sasa_metrics(sasa_weighted)
            overall_flexibility = compute_overall_flexibility(flexibility_weighted)
            glycan_class = get_class(matched_glycan) or np.nan

            metric_data.append({
                "glycan": glycan,
                "binding_score": binding_score,
                "SASA_weighted": sasa_metrics["SASA_weighted"],
                "SASA_weighted_max": sasa_metrics.get("SASA_weighted_max"),
                "SASA_weighted_sum": sasa_metrics.get("SASA_weighted_sum"),
                "weighted_mean_flexibility": overall_flexibility,
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
    metric_df.set_index('glycan', inplace=True)
#    metric_df.to_excel(f'scripts/correlation/metric_df/{lectin}_metrics.xlsx', index=True, header=True)
    return metric_df

#sys.stdout = open('scripts/correlation/metric_df/metric_df.log', 'w')

def perform_mediation_analysis_with_class(metric_df, independent_var, class_var, dependent_var):
    """
    Perform mediation analysis to test if glycan class mediates the relationship between
    the independent variable and the dependent variable. Handles NaN values and single-class cases.

    Parameters:
        metric_df (pd.DataFrame): The DataFrame containing metrics for mediation analysis.
        independent_var (str): Name of the independent variable (e.g., 'weighted_mean_flexibility').
        class_var (str): Name of the mediator variable (categorical, e.g., 'class').
        dependent_var (str): Name of the dependent variable (e.g., 'binding_score').

    Returns:
        dict: A dictionary containing mediation results and total, direct, and indirect effects,
              or a message indicating mediation cannot be performed.
    """

    # Step 1: Drop rows with NaN in the class column
    metric_df = metric_df.dropna(subset=[class_var]).copy()

    # Step 2: Check the number of unique classes in the mediator column
    unique_classes = metric_df[class_var].unique()
    if len(unique_classes) < 2:
        return {
            'message': f"Mediation analysis cannot be performed. Found only one unique class: {unique_classes}."
        }

    # Step 3: Encode class as binary (0 and 1) using LabelEncoder
    label_encoder = LabelEncoder()
    metric_df.loc[:, 'encoded_class'] = label_encoder.fit_transform(metric_df[class_var])

    # Step 4: Perform mediation analysis
    mediation_results = pg.mediation_analysis(
        data=metric_df,
        x=independent_var,  # Independent variable
        m='encoded_class',  # Encoded mediator (binary glycan class)
        y=dependent_var,  # Dependent variable
        alpha=0.05
    )

    # Step 5: Extract mediation effects
    total_effect = mediation_results.loc[mediation_results['path'] == 'Total', 'coef'].values[0]
    direct_effect = mediation_results.loc[mediation_results['path'] == 'Direct', 'coef'].values[0]
    indirect_effect = mediation_results.loc[mediation_results['path'] == 'Indirect', 'coef'].values[0]

    return {
        'results': mediation_results,
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect
    }



""" Visualisation"""

import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import networkx as nx
import os

sns.set(style="whitegrid")
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = '--'
plt.figure(figsize=(8, 6))

def plot_separate_class(metric_df, lectin, binding_motif):
    """Plots Binding vs Flexibility and Binding vs SASA Weighted for N- and O-linked glycans in separate rows."""

    # Filter for O-glycans and N-glycans
    o_glycans = metric_df[metric_df['class'] == 'O']
    n_glycans = metric_df[metric_df['class'] == 'N']

    # Create a single figure with two rows and two columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # 2 rows for O and N, 2 columns for Flexibility and SASA

    # Define plotting function for individual glycan class
    def plot_single_metric(data, glycan_type, axis_flex, axis_sasa):
        # Binding vs Flexibility
        scatter_flex = sns.scatterplot(
            ax=axis_flex,
            x='weighted_mean_flexibility',
            y='binding_score',
            data=data,
            hue="class",
            hue_order=['N', 'O', "free", "lipid", "lipid/free" , ""],  # Ensure consistent color scheme
            palette="tab10",
            alpha=0.7,
            legend=False  # Suppress legend for this plot
        )
        sns.regplot(
            ax=axis_flex,
            x='weighted_mean_flexibility',
            y='binding_score',
            data=data,
            scatter=False,
            line_kws={'color': 'red'}
        )
        axis_flex.set_title(f'{glycan_type} Glycans: Binding vs Flexibility Weighted\n{lectin} {binding_motif}', fontsize=12)
        axis_flex.set_xlabel('Flexibility')
        axis_flex.set_ylabel('Binding Score')

        # Binding vs SASA
        scatter_sasa = sns.scatterplot(
            ax=axis_sasa,
            x='SASA_weighted',
            y='binding_score',
            data=data,
            hue='class',
            hue_order=['N', 'O', "free", "lipid", "lipid/free" , ""],  # Ensure consistent color scheme
            palette="tab10",
            alpha=0.7,
            legend=False  # Suppress legend for this plot
        )
        sns.regplot(
            ax=axis_sasa,
            x='SASA_weighted',
            y='binding_score',
            data=data,
            scatter=False,
            line_kws={'color': 'red'}
        )
        axis_sasa.set_title(f'{glycan_type} Glycans: Binding vs SASA Weighted\n{lectin} {binding_motif}', fontsize=12)
        axis_sasa.set_xlabel('SASA Weighted')
        axis_sasa.set_ylabel('Binding Score')

        # Return handles and labels for legend
        handles, labels = scatter_sasa.get_legend_handles_labels()
        return handles, labels

    # Row 1: O-linked glycans
    handles_o, labels_o = plot_single_metric(o_glycans, 'O-linked', axes[0, 0], axes[0, 1])

    # Row 2: N-linked glycans
    handles_n, labels_n = plot_single_metric(n_glycans, 'N-linked', axes[1, 0], axes[1, 1])

    # Create a single unified legend using one of the scatterplots
    fig.legend(
        handles=handles_o[:5],  # Use up to 5 classes for consistency
        labels=labels_o[:5],
        title="Glycan class",
        bbox_to_anchor=(1.02, 0.5),  # Position legend closer to the plot
        loc='center left',
        borderaxespad=0  # Reduce padding
    )

    # Adjust layout to make space for the legend
    plt.subplots_adjust(right=0.8)  # Adjust the right margin to make space for the legend

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'results/Binding_vs_Flexibility_and_SASA_{lectin}_O_N_separated.png', dpi=300)

    # Show the plots
    plt.show()

def plot_mediation_results_with_class(metric_df,
                                           independent_var,
                                           class_var,
                                           dependent_var,
                                           effects,
                                           lectin,
                                           binding_motif):
    """
    Visualize mediation analysis results with a path diagram, scatterplots, and a bar chart.

    Parameters:
        metric_df (pd.DataFrame): The data containing variables for analysis.
        independent_var (str): Name of the independent variable.
        class_var (str): Name of the mediator variable (categorical, e.g., 'class').
        dependent_var (str): Name of the dependent variable.
        effects (dict): Dictionary containing total, direct, and indirect effects.
        lectin (str): Name of the lectin being analyzed.
        binding_motif (str): Name of the binding motif being analyzed.
        save_dir (str): Directory where plots will be saved. Defaults to the current directory.
    """
    save_dir = f"results/{independent_var}/"
    os.makedirs(save_dir, exist_ok=True)

    # Unpack effects
    total_effect = effects['total_effect']
    direct_effect = effects['direct_effect']
    indirect_effect = effects['indirect_effect']

    # Step 1: Path Diagram
    def plot_mediation_path():
        G = nx.DiGraph()
        G.add_edge(independent_var, 'Class (Mediator)', weight=indirect_effect)
        G.add_edge('Class (Mediator)', dependent_var, weight=indirect_effect)
        G.add_edge(independent_var, dependent_var, weight=direct_effect)

        pos = {
            independent_var: (0, 1),
            'Class (Mediator)': (1, 0.5),
            dependent_var: (2, 1)
        }

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
        edges = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={k: f"{v:.3f}" for k, v in edges.items()}, font_color='red'
        )
        plt.title(f"Mediation Path Diagram\nLectin: {lectin}, Motif: {binding_motif}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{lectin}_{binding_motif}_mediation_path.png"))
        plt.show()

    # Step 2: Scatterplots
    def plot_scatterplots():
        plt.figure(figsize=(12, 6))

        # Define color palette for classes
        custom_palette = {"N": "blue", "O": "orange", "free": "green", "lipid": "red", "lipid/free": "purple", "": "gray"}

        # Independent vs Class (Mediator)
        plt.subplot(1, 3, 1)
        sns.boxplot(x=class_var, y=independent_var, data=metric_df, palette=custom_palette)
        plt.title(f"{independent_var} vs. {class_var}\nLectin: {lectin}, Motif: {binding_motif}")
        plt.xlabel(class_var)
        plt.ylabel(independent_var)

        # Class (Mediator) vs Dependent
        plt.subplot(1, 3, 2)
        sns.boxplot(x=class_var, y=dependent_var, data=metric_df, palette=custom_palette)
        plt.title(f"{class_var} vs. {dependent_var}\nLectin: {lectin}, Motif: {binding_motif}")
        plt.xlabel(class_var)
        plt.ylabel(dependent_var)

        # Independent vs Dependent
        plt.subplot(1, 3, 3)
        sns.regplot(x=independent_var, y=dependent_var, data=metric_df, scatter_kws={'color': 'blue'})
        plt.title(f"{independent_var} vs. {dependent_var}\nLectin: {lectin}, Motif: {binding_motif}")
        plt.xlabel(independent_var)
        plt.ylabel(dependent_var)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{lectin}_{binding_motif}_scatterplots.png"))
        plt.show()

    # Step 3: Bar Chart for Effects
    def plot_effects_bar_chart():
        effects_labels = ['Total Effect', 'Direct Effect', 'Indirect Effect']
        effects_values = [total_effect, direct_effect, indirect_effect]
        plt.figure(figsize=(8, 6))
        plt.bar(effects_labels, effects_values, color=['blue', 'green', 'orange'])
        plt.title(f"Mediation Effects\nLectin: {lectin}, Motif: {binding_motif}")
        plt.ylabel("Effect Value")
        plt.savefig(os.path.join(save_dir, f"{lectin}_{binding_motif}_effects_bar_chart.png"))
        plt.show()

    # Call visualizations
    print("\nStep 1: Mediation Path Diagram")
    plot_mediation_path()

    print("\nStep 2: Scatterplots")
    plot_scatterplots()

    print("\nStep 3: Mediation Effects Bar Chart")
    plot_effects_bar_chart()

def plot_Binding_vs_Flexibility_and_SASA_with_stats(metric_df, lectin, binding_motif):
    """Plots Binding vs Flexibility and Binding vs SASA weighted with colors for glycans,
    and annotates the plots with formal regression coefficients and associated p-values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Create two subplots side by side

    # -------------------------------
    # Plot 1: Binding vs Flexibility
    # -------------------------------
    # Scatter plot colored by glycan class
    sns.scatterplot(
        ax=axes[0],
        x='weighted_mean_flexibility',
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
        x='weighted_mean_flexibility',
        y='binding_score',
        data=metric_df,
        scatter=False,  # Do not plot points again
        line_kws={'color': 'red'}
    )
    axes[0].set_title(f'Binding vs Flexibility Weighted\n{lectin} {binding_motif}', fontsize=12)
    axes[0].set_xlabel('Flexibility')
    axes[0].set_ylabel('Binding Score')
    axes[0].get_legend().remove()  # Remove legend from the first plot

    # Compute regression parameters for Binding vs Flexibility
    flex = metric_df['weighted_mean_flexibility']
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
        x='SASA_weighted',
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
        x='SASA_weighted',
        y='binding_score',
        data=metric_df,
        scatter=False,  # Do not plot points again
        line_kws={'color': 'red'}
    )
    axes[1].set_title(f'Binding vs SASA Weighted\n{lectin} {binding_motif}', fontsize=12)
    axes[1].set_xlabel('SASA Weighted')
    axes[1].set_ylabel('Binding Score')
    axes[1].legend(title="Glycan class", bbox_to_anchor=(1.05, 1), loc='upper left')  # Keep class legend

    # Compute regression parameters for Binding vs SASA
    sasa = metric_df['SASA_weighted']
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
    plt.savefig(f'results/Binding_vs_Flexibility_and_SASA_{lectin}.png', dpi=300)

    # Show the plots
    plt.show()
