from inspect import unwrap
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.graph import glycan_to_nxGraph
from pandas.core.interchange.dataframe_protocol import DataFrame

from scripts.load_data import load_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

lectin_binding_motif = {
    "AOL": {"motif": ["Fuc(a1-?)"],
            "termini_list": [["t"]]},
    "AAL": {
        "motif": ["Fuc(a1-?)"],
        "termini_list": [["t"]]
    },
    "SNA": {
        "motif": ["Sia(a2-6)"],
        "termini_list": [["t"]]
    },
    "ConA": {
        "motif": ["Man(a1-?)"],
        "termini_list": [["t"]]
    },
    "MAL-II": {
        "motif": ["Sia(a2-3)"],
        "termini_list": [["t"]]
    },
    "PNA": {
        "motif": ["Gal(b1-3)GalNAc"],
        "termini_list": [["t", "f"]]
    },
    "CMA": {
        "motif": ["Fuc(a1-2)Gal", "GalNAc"],
        "termini_list": [["t", "f"], ["t"]]
    },
    "HPA": {
        "motif": ["GalNAc(a1-?)", "GlcNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },

    "AMA": {
        "motif": ["GlcNAc(b1-2/4)Man(a1-?)", "Man(a1-2)"],
        "termini_list": [["t", "f"], ["t"]]
    },
    "GNA": {
        "motif": ["Man(a1-6)", "Man(a1-3)"],
        "termini_list": [["t"], ["t"]]
    },
    "HHL": {
        "motif": ["Man(a1-?)"],
        "termini_list": [["t"]]
    },
    "MNA-M": {
        "motif": ["Man(a1-3)", "Man(a1-6)", "GlcNAc(b1-?)Man(a1-?)"],
        "termini_list": [["t"], ["t"], ["t", "f"]]
    },
    "NPA": {
        "motif": ["Man(a1-6)", "Man(a1-3)"],
        "termini_list": [["t"], ["t"]]
    },
    "UDA": {
        "motif": ["Man(a1-6)"],
        "termini_list": [["t"]]
    },
    "ABA": {
        "motif": ["GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man"],
        "termini_list": [["f", "f", "f", "f", "f"]]
    },
    "CA": {
        "motif": ["Gal(b1-3/4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-3/4)GlcNAc(b1-2)Man(a1-6)]Man"],
        "termini_list": [["f", "f", "f", "f", "f", "f", "f"]]
    },
    "CAA": {
        "motif": ["Gal(b1-3/4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-3/4)GlcNAc(b1-2)Man(a1-6)]Man"],
        "termini_list": [["f", "f", "f", "f", "f", "f", "f"]]
    },
    "TL": {
        "motif": ["GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"],
        "termini_list": [["f", "f", "f", "f", "f", "f", "f", "f"]]
    },
    "ACA": {
        "motif": ["Gal(b1-3)GalNAc", "GalOS(b1-3)GalNAc"],
        "termini_list": [["f", "f"], ["f", "f"]]
    },

    "AIA": {
        "motif": ["Gal(b1-3)GalNAc", "GlcNAc(b1-3)GalNAc"],
        "termini_list": [["f", "f"], ["f", "f"]]
    },
    "CF": {
        "motif": ["GalNAc(a1-?)", "GlcNAc(a1-?)"],
        "termini_list": [["f"], ["f"]]
    },
    "HAA": {
        "motif": ["GalNAc(a1-?)", "GlcNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },
    "MPA": {
        "motif": ["Gal(b1-3)GlcNAc", "GlcNAc(b1-3)GalNAc"],
        "termini_list": [["f", "f"], ["f", "f"]]
    },
    "LAA": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f", "f"]]
    },
    "LcH": {
        "motif": ["Fuc(a1-6)", "Man(a1-2)"],
        "termini_list": [["t"], ["t"]]
    },
    "LTL": {
        "motif": ["Fuc(a1-3)[Gal(b1-4)]GlcNAc", ],
        "termini_list": [["t", "f", "f"]]
    },
    "LTA": {
        "motif": ["Fuc(a1-3)[Gal(b1-4)]GlcNAc", ],
        "termini_list": [["t", "f", "f"]]
    },
    "PSA": {
        "motif": ["Fuc(a1-6)"],
        "termini_list": [["t"]]
    },

    "PTL-I": {
        "motif": ["Fuc(a1-2)[GalNAc(a1-3)]Gal", "Fuc(a1-2)[Gal(a1-3)]Gal"],
        "termini_list": [["t", "t", "f"], ["t", "t", "f"]] #double t?
        #"termini_list": [["t", "f", "f"], ["t", "f", "f"]]  # double t?

    },

    "PTA-I": {
        "motif": ["Fuc(a1-2)[GalNAc(a1-3)]Gal", "Fuc(a1-2)[Gal(a1-3)]Gal"],
        "termini_list": [["t", "t", "f"], ["t", "t", "f"]]
      #  "termini_list": [["t", "f", "f"], ["t", "f", "f"]]  # double t?

    },

    "PTA-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f", "f"]]
    },
    "PTL-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f", "f"]]
    },

    "TJA-II": {
        "motif": ["Fuc(a1-2)Gal(b1-3)GalNAc", "Fuc(a1-2)Gal(b1-3/4)GlcNAc"],
        "termini_list": [["t", "f", "f"], ["t", "f", "f"]]
    },
    "UEA-I": {
        "motif": ["Fuc(a1-2)Gal(b1-4)"],
        "termini_list": [["t", "f"]]
    },

    "CTB": {
        "motif": ["Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)",
                  "Fuc(a1-2)Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)"],
        "termini_list": [["t", "f", "t", "f", "f"], ["t", "f", "f", "t", "f", "f"]]

    },

    "MAL-I": {
        "motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
        #"motif": ["Gal(b1-4)GlcNAc", "Gal(b1-4)GlcNAc6S", "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },

    "MAA": {
        #"motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
        "motif": ["Gal(b1-4)GlcNAc", "Gal(b1-4)GlcNAc6S", "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },
    "MAL": {
        "motif": ["Gal(b1-4)GlcNAc", "Gal(b1-4)GlcNAc6S", "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"],
        #"motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },

    "PSL": {
        "motif": ["Sia(a2-6)Gal(b1-3/4)GlcNAc"],
        "termini_list": [["t", "f", "f"]]
    },
    "TJA-I": {
        "motif": ["Sia(a2-6)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f", "f"]]
    },

    "GS-II": {
        "motif": ["GlcNAc"],
        "termini_list": [["t"]]
    },
    "PWA": {
        "motif": ["GlcNAc(b1-4)GlcNAc(b1-4)GlcNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "f", "f", "f"]]
    },
    "UEA-II": {
        "motif": ["GlcNAc(b1-3)", "Fuc(a1-2)Gal(b1-4)GlcNAc", "Fuc(a1-2)Gal(b1-3)GalNAc"],
        "termini_list": [["t"], ["t", "f", "f"], ["t", "f", "f"]]
    },
    "WGA": {
        "motif": ["GlcNAc", "GalNAc", "Sia(a2-?)", "MurNAc(b1-?)"],
        "termini_list": [["t"], ["t"], ["t"], ["t"]]
    },

    "BPA": {
        "motif": ["Gal(b1-?)", "GalNAc(b1-?)"],
        "termini_list": [["t"], ["t"]]
    },
    "BPL": {
        "motif": ["Gal(b1-?)", "GalNAc(b1-?)"],
        "termini_list": [["t"], ["t"]]
    },

    "ECA": {
        "motif": ["Gal(b1-4)GlcNAc", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"]]
    },
    "GS-I": {
        "motif": ["Gal(a1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },

    "LEA": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)GlcNAc", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f"]]
    },
    "LEL": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)GlcNAc", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f"]]
    },

    "MOA": {
        "motif": ["Gal(a1-3)Gal", "Gal(a1-3)GalNAc"],
        "termini_list": [["t", "f"], ["t", "f"]]
    },
    "PA-IL": {
        "motif": ["Gal(a1-?)"],
        "termini_list": [["t"]]
    },
    "LecA": {
        "motif": ["Gal(a1-?)"],
        "termini_list": [["t"]]
    },
    "RCA-I": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },
    "RCA120": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },

    "SJA": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)", "Fuc(a1-2)[Gal(a1-3)]Gal", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "f", "f", "f"], ["t", "t", "f"], ["t", "f"]]
    },

    "STA": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)GlcNAc", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f"]]
    },
    "STL": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)GlcNAc", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f"]]
    },

    "CSA": {
        "motif": ["GalNAc"],
        "termini_list": [["t"]]
    },
    "DBA": {
        "motif": ["GalNAc(a1-3)GalNAc(b1-?)"],
        "termini_list": [["t", "f"]]
    },
    "SBA": {
        "motif": ["GalNAc", "GalNAc(a1-3)GalNAc(b1-?)"],
        "termini_list": [["t"], ["t", "f"]]
    },

    "VVL": {
        "motif": ["GalNAc"],
        "termini_list": [["t"]]
    },
    "VVA": {
        "motif": ["GalNAc"],
        "termini_list": [["t"]]
    },

    "WFA": {
        "motif": ["GalNAc", "Gal(b1-3/4)GlcNAc(b1-3)"],
        "termini_list": [["t"], ["t", "f"]]
    }
}
lectin_keys= {
    "LTL": "LTL",
    "PTL-I": "PTL-I",
    "MAL": "MAL-I",
    "BPL": "BPL",
    "LEL": "LEL",
    "STL": "STL",
    "VVL": "VVL",
    "AOL": "AOL",
    "AAA": "AAA",
    "AAL": "AAL",
    "SNA": "SNA",
    "SNA-I": "SNA",
    "ConA": "ConA",
    "MAL-II": "MAL-II",
    "PNA": "PNA",
    "CMA": "CMA",
    "HPA": "HPA",
    "AMA": "AMA",
    "GNA": "GNA",
    "GNL": "GNA",
    "HHL": "HHL",
    "MNA-M": "MNA-M",
    "Morniga": "MNA-M",
    "NPA": "NPA",
    "NPL": "NPA",
    "UDA": "UDA",
    "ABA": "ABA",
    "CA": "CA",
    "CAA": "CAA",
    "TL": "TL",
    "ACA": "ACA",
    "AIA": "AIA",
    "Jacalin": "AIA",
    "CF": "CF",
    "HAA": "HAA",
    "MPA": "MPA",
    "LAA": "LAA",
    "LcH": "LcH",
    "LCA": "LcH",
    "LTA": "LTA",
    "LTL": "LTA",
    "PSA": "PSA",
    "PTA-I": "PTL-I",
    "PTA-II": "PTL-II",
    "PTL-II": "PTL-II",
    "TJA-II": "TJA-II",
    "UEA-I": "UEA-I",
    "CTB": "CTB",
    "MAL-I": "MAL-I",
    "MAA": "MAL-I",
    "PSL": "PSL",
    "TJA-I": "TJA-I",
    "GS-II": "GS-II",
    "PWA": "PWA",
    "UEA-II": "UEA-II",
    "WGA": "WGA",
    "BPA": "BPL",
    "ECA": "ECL",
    "GS-I": "GS-I",
    "GSL-I": "GS-I",
    "LEA": "LEL",
    "MOA": "MOA",
    "PA-IL": "PA-IL",
    "LecA": "PA-IL",
    "RCA-I": "RCA-I",
    "RCA120": "RCA-I",
    "SJA": "SJA",
    "STA": "STL",
    "CSA": "CSA",
    "DBA": "DBA",
    "SBA": "SBA",
    "VVA": "VVL",
    "WFA": "WFL",
    "WFL": "WFL"
}

structure_graphs, glycan_binding, invalid_graphs = load_data()

lectins_filt = {k: v for k, v in lectin_binding_motif.items()
                if any(
    len(t) > 1 for t in v['termini_list'])
                and k in glycan_binding.protein.tolist()}  # compare_aggregations

lectins = {k: v for k, v in lectin_binding_motif.items() if k in glycan_binding.protein.tolist()}


binding_data = glycan_binding.set_index('protein').drop(['target'], axis=1).T #subset of glycan_binding

binding_data = binding_data.rename(columns=lectin_keys)  # synonym dict
binding_data = binding_data.T.groupby(level=0).median().T


glycan_dict = {g: v for g, v in structure_graphs.items() if
               g in binding_data.index or any(compare_glycans(g, b) for b in binding_data.index)}



def plot_correlation_boxplot_(sasa_all, flex_all, title="Correlation Distributions"):
    """
    Creates a box plot showing the distributions of correlation values alongside their median values.

    Parameters:
    -----------
    sasa_all : dict
        Dictionary where keys are aggregation method names and values are DataFrames/Series of SASA correlations
    flex_all : dict
        Dictionary where keys are aggregation method names and values are DataFrames/Series of flexibility correlations
    title : str, optional
        Title for the plot

    Returns:
    --------
    fig, ax : tuple
        The figure and axes objects for further customization if needed
    """
    # Prepare data for plotting
    plot_data = []
    labels = []
    medians = []
    colors = []

    # Process SASA correlations
    for name, corr_data in sasa_all.items():
        # Convert to absolute values and flatten if it's a DataFrame
        if isinstance(corr_data, pd.DataFrame):
            values = corr_data.abs().values.flatten()
        else:
            values = corr_data.abs().values

        # Remove NaN values
        values = values[~np.isnan(values)]

        plot_data.append(values)
        labels.append(f"SASA-{name}")
        medians.append(np.median(values))
        colors.append('lightblue')

    # Process flexibility correlations
    for name, corr_data in flex_all.items():
        # Convert to absolute values and flatten if it's a DataFrame
        if isinstance(corr_data, pd.DataFrame):
            values = corr_data.abs().values.flatten()
        else:
            values = corr_data.abs().values

        # Remove NaN values
        values = values[~np.isnan(values)]

        plot_data.append(values)
        labels.append(f"Flex-{name}")
        medians.append(np.median(values))
        colors.append('lightgreen')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create box plot
    bp = ax.boxplot(plot_data, patch_artist=True, showfliers=True, widths=0.6)

    # Customize box plot colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add scatter points for the actual data (jittered)
    for i, data in enumerate(plot_data):
        # Add jitter to x position
        x = np.random.normal(i + 1, 0.08, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=10, c='darkgray', edgecolor='none')

    # Add markers for the median values
    for i, median in enumerate(medians):
        ax.scatter(i + 1, median, s=100, c='red', marker='*',
                   label='Median' if i == 0 else "", zorder=3)

    # Add a legend for the median marker (only once)
    ax.legend()

    # Customize the plot
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Absolute Correlation Value', fontsize=12)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add a grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Tight layout to ensure labels are visible
    plt.tight_layout()

    return fig, ax

def get_correlations_(lectins: list , glycan_dict, binding_data: DataFrame, agg1, agg2):
    sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())

    def safe_agg(func, data):
        if len(data) == 0:
            return np.nan  # Return NaN for empty arrays
        return func(data)

    for lectin in lectins:
        print(lectin)
        binding_motif = lectin_binding_motif[lectin]
        motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided',
                                          termini_list=binding_motif['termini_list'][i]) for i in
                        range(len(binding_motif['motif']))]
        all_sasa, all_flex = [], []
        for _, ggraph in glycan_dict.items():
            _, matches = zip(*[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])

            # Use safe_agg to handle empty arrays
            sasa_values = [safe_agg(agg1, [ggraph.nodes()[n].get('SASA', np.nan) for n in m]) for m in
                           unwrap(matches[0])]
            all_sasa.append(safe_agg(agg2, sasa_values))

            flex_values = [safe_agg(agg1, [ggraph.nodes()[n].get('flexibility', np.nan) for n in m]) for m in
                           unwrap(matches[0])]
            all_flex.append(safe_agg(agg2, flex_values))

        sasa_df[lectin] = all_sasa
        flex_df[lectin] = all_flex

    sasa_corr = sasa_df.corrwith(binding_data, axis=0, drop=False, method='pearson')
    sasa_corr = sasa_corr.reset_index().groupby("index").median() # agg multiple proteins columns in binding data
    #sasa_corr = sasa_corr.dropna()  # drop lectins with no corr

    flex_corr = flex_df.corrwith(binding_data)  # drop lectins with no corr
    flex_corr = flex_corr.reset_index().groupby("index").median()
    #flex_corr = flex_corr.dropna()
    return sasa_corr, flex_corr
def get_correlations__(lectins: list, glycan_dict, binding_data: DataFrame, agg1, agg2):
    sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())

    for lectin in lectins:
        print(lectin)
        binding_motif = lectin_binding_motif[lectin]
        motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided',
                                          termini_list=binding_motif['termini_list'][i]) for i in
                        range(len(binding_motif['motif']))]
        all_sasa, all_flex = [], []

        for glycan_key, ggraph in glycan_dict.items():
            _, matches = zip(
                *[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])

            # Collect all matches from all motif graphs
            all_matches = []
            for match_set in matches:
                all_matches.extend(unwrap(match_set))

            # Process SASA values with defaults for error cases
            sasa_values = []
            for match in all_matches:
                node_sasa_values = [ggraph.nodes[n].get('SASA', np.nan) for n in match]
                if node_sasa_values and not all(np.isnan(v) for v in node_sasa_values):
                    sasa_values.append(agg1(node_sasa_values))

            # Process flexibility values with defaults for error cases
            flex_values = []
            for match in all_matches:
                node_flex_values = [ggraph.nodes[n].get('flexibility', np.nan) for n in match]
                if node_flex_values and not all(np.isnan(v) for v in node_flex_values):
                    flex_values.append(agg1(node_flex_values))

            # Apply second aggregation with proper defaults
            all_sasa.append(agg2(sasa_values) if sasa_values else np.nan)
            all_flex.append(agg2(flex_values) if flex_values else np.nan)

        sasa_df[lectin] = all_sasa
        flex_df[lectin] = all_flex

    sasa_corr = sasa_df.corrwith(binding_data, axis=0, drop=False, method='pearson').reset_index().groupby(
        "index").median()
    flex_corr = flex_df.corrwith(binding_data).reset_index().groupby("index").median()

    return sasa_corr, flex_corr
def get_correlations___(lectins, glycan_dict, binding_data, agg1=np.nansum, agg2=np.nanmean):
    sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())

    for lectin in lectins:
        binding_motif = lectin_binding_motif[lectin]
        motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided',
                                          termini_list=binding_motif['termini_list'][i])
                        for i in range(len(binding_motif['motif']))]
        all_sasa, all_flex = [], []

        for glycan_key, ggraph in glycan_dict.items():
            _, matches = zip(*[subgraph_isomorphism(ggraph, mg, return_matches=True) for mg in motif_graphs])
            all_matches = [m for match_set in matches for m in unwrap(match_set)]

            # Handle empty match cases and apply safe aggregation
            sasa_values = [
                agg1([val for val in [ggraph.nodes[n].get('SASA', np.nan) for n in match] if not np.isnan(val)])
                if any(not np.isnan(ggraph.nodes[n].get('SASA', np.nan)) for n in match)
                else np.nan
                for match in all_matches]
            flex_values = [agg1([ggraph.nodes[n].get('flexibility', np.nan) for n in match]) for match in all_matches]

            all_sasa.append(agg2(sasa_values) if sasa_values else np.nan)
            all_flex.append(agg2(flex_values) if flex_values else np.nan)

        sasa_df[lectin] = all_sasa
        flex_df[lectin] = all_flex

    # Match original post-processing
    sasa_corr = sasa_df.corrwith(binding_data, axis=0, drop=False, method='pearson').reset_index().groupby(
        "index").median()
    flex_corr = flex_df.corrwith(binding_data).reset_index().groupby("index").median()

    return sasa_corr, flex_corr
def get_correlations4(lectins, glycan_dict, binding_data, agg1=np.nansum, agg2=np.nanmean):
    # Initialize DataFrames for SASA and flexibility values
    sasa_df = pd.DataFrame(index=glycan_dict.keys())
    flex_df = pd.DataFrame(index=glycan_dict.keys())

    for lectin in lectins:
        # Get binding motif information for current lectin
        binding_motif = lectin_binding_motif[lectin]

        # Create motif graphs
        motif_graphs = [
            glycan_to_nxGraph(
                binding_motif['motif'][i],
                termini='provided',
                termini_list=binding_motif['termini_list'][i]
            ) for i in range(len(binding_motif['motif']))
        ]

        all_sasa = []
        all_flex = []

        # Process each glycan for the current lectin
        for glycan_key, ggraph in glycan_dict.items():
            # Find all matches across motif graphs
            iso_results = [subgraph_isomorphism(ggraph, mg, return_matches=True) for mg in motif_graphs]
            _, matches = zip(*iso_results)

            # Flatten all matches into a single list
            all_matches = [m for match_set in matches for m in unwrap(match_set)]

            # Calculate SASA values with proper NaN handling
            sasa_values = []
            for match in all_matches:
                node_sasa_values = [ggraph.nodes[n].get('SASA', np.nan) for n in match]
                # Filter out NaN values before aggregation if at least one valid value exists
                valid_sasa_values = [v for v in node_sasa_values if not np.isnan(v)]
                if valid_sasa_values:
                    sasa_values.append(agg1(valid_sasa_values))
                else:
                    sasa_values.append(np.nan)

            # Calculate flexibility values with proper NaN handling
            flex_values = []
            for match in all_matches:
                node_flex_values = [ggraph.nodes[n].get('flexibility', np.nan) for n in match]
                # Filter out NaN values before aggregation if at least one valid value exists
                valid_flex_values = [v for v in node_flex_values if not np.isnan(v)]
                if valid_flex_values:
                    flex_values.append(agg1(valid_flex_values))
                else:
                    flex_values.append(np.nan)

            # Aggregate match values for this glycan
            glycan_sasa = agg2(sasa_values) if sasa_values else np.nan
            glycan_flex = agg2(flex_values) if flex_values else np.nan

            all_sasa.append(glycan_sasa)
            all_flex.append(glycan_flex)

        # Store results for current lectin
        sasa_df[lectin] = all_sasa
        flex_df[lectin] = all_flex

    # Calculate correlations
    sasa_corr = sasa_df.corrwith(binding_data, axis=0, drop=False, method='pearson')
    sasa_corr_reset = sasa_corr.reset_index()
    sasa_corr_final = sasa_corr_reset.groupby("index").median()

    flex_corr = flex_df.corrwith(binding_data)
    flex_corr_reset = flex_corr.reset_index()
    flex_corr_final = flex_corr_reset.groupby("index").median()

    return sasa_corr_final, flex_corr_final



def get_correlations5(lectins, glycan_dict, binding_data, agg1, agg2):
    sdf = pd.DataFrame(index=glycan_dict.keys())
    fdf = pd.DataFrame(index=glycan_dict.keys())

    for lec in lectins:
        bmot = lectin_binding_motif[lec]
        mg = []
        for i in range(len(bmot['motif'])):
            mot = bmot['motif'][i]
            term_l = bmot['termini_list'][i]
            g = glycan_to_nxGraph(mot, termini='provided', termini_list=term_l)
            mg.append(g)


        sasa_dict = {}
        flex_dict = {}
        for gkey, gg in glycan_dict.items():
            # Collect all matches from different motifs
            matches = []
            for mot in mg:
                res = subgraph_isomorphism(gg, mot, return_matches=True)
                ms = res[1]
                matches.extend(unwrap(ms))
            if not matches:
                continue

            s_vals = []
            f_vals = []
            for m in matches:
                # Calculate values for this match
                s = agg1([gg.nodes[n].get('SASA', np.nan) for n in m])
                f = agg1([gg.nodes[n].get('flexibility', np.nan) for n in m])
                s_vals.append(s)
                f_vals.append(f)

            # Use single value or aggregate based on count
            s_res = agg2(s_vals) if len(s_vals) > 1 else (s_vals[0] if s_vals else np.nan)
            f_res = agg2(f_vals) if len(f_vals) > 1 else (f_vals[0] if f_vals else np.nan)

            # Store in dictionaries
            sasa_dict[gkey] = s_res
            flex_dict[gkey] = f_res

        # Assign dictionaries to columns
        sdf[lec] = pd.Series(sasa_dict)
        fdf[lec] = pd.Series(flex_dict)

    # Calculate correlations
    s_corr = sdf.corrwith(binding_data, axis=0, drop=False, method='pearson')
    s_final = s_corr.reset_index().groupby("index").median()

    f_corr = fdf.corrwith(binding_data)
    f_final = f_corr.reset_index().groupby("index").median()

    return s_final, f_final
"""
Lectin-Binding Motif > 1 mono
agg1 = variable #sum/mean won!
agg2 = fixed


agg_list = [np.nansum, np.nanmax, np.nanmean]
agg_n = ['nansum', 'nanmax', 'nanmean']


sasa_filt = {}
flex_filt = {}
x = list(glycan_dict.keys())
y = list(set(list(lectin_keys.values()))) # set to remove duplicates
found_lectins = [lectin for lectin in y if lectin in binding_data.columns]


for i, name in zip(agg_list, agg_n):
    sasa_corr, flex_corr= get_correlations5(
        lectins_filt.keys(), #subest of lectins
        glycan_dict,
        #binding_data.loc[x, binding_data.columns],
        binding_data.loc[x, found_lectins],
        agg1=i,
        agg2=np.nansum)

    sasa_filt[name] = sasa_corr
    flex_filt[name] = flex_corr

    print(name)
    sasa_filt_abs_median = float(sasa_corr.abs().median())
    print(f"sasa_filt_abs_median : {sasa_filt_abs_median:.2f}")
    flex_filt_abs_median = float(flex_corr.abs().median())
    print(f"flex_filt_abs_median: {flex_filt_abs_median:.2f}")
    print("")

plot_correlation_boxplot(sasa_filt, flex_filt, "Correlation Distribution Filtered Lectins")
plt.savefig("results/plots/correlation_boxplot_filtered_lectins.png", dpi=300)
plt.show()


all Lectins
agg1 = fixed
agg2 = variable #max won!



sasa_all = {}
flex_all = {}

for i, name in zip(agg_list, agg_n):
    sasa_corr, flex_corr = get_correlations5(
        lectins.keys(),
        glycan_dict,
        #binding_data.loc[list(glycan_dict.keys()), binding_data.columns],
        binding_data.loc[x, found_lectins],
        agg1=np.nansum,
        agg2=i)

    sasa_all[name] = sasa_corr
    flex_all[name] = flex_corr


    print(name)
    sasa_all_abs_median = float(sasa_corr.abs().median()    )
    print(f"sasa_all_abs_median: {sasa_all_abs_median:.2f}")
    flex_all_abs_median = float (flex_corr.abs().median()   )
    print(f"flex_all_abs_median: {flex_all_abs_median:.2f}")
    print("")

plot_correlation_boxplot(sasa_all, flex_all, "Correlation Distribution all Lectins")
plt.savefig("results/plots/correlation_boxplot_all_lectins.png", dpi=300)
plt.show()
"""


"""
Common setup for both lectin sets
"""
agg_list = [np.nansum, np.nanmax, np.nanmean]
agg_n = ['nansum', 'nanmax', 'nanmean']
x = list(glycan_dict.keys())
y = list(set(list(lectin_keys.values())))  # set to remove duplicates

# For filtered lectins
filt_valid_lectins = [lectin for lectin in lectins_filt.keys() if lectin in binding_data.columns]
# For all lectins
all_valid_lectins = [lectin for lectin in lectins.keys() if lectin in binding_data.columns]

"""
Filtered Lectins
agg1 = variable (sum/mean/max)
agg2 = fixed (nansum)
"""
sasa_filt = {}
flex_filt = {}

for i, name in zip(agg_list, agg_n):
    sasa_corr, flex_corr = get_correlations5(
        filt_valid_lectins,  # Only use lectins that exist in both datasets
        glycan_dict,
        binding_data.loc[x, filt_valid_lectins],  # Use same lectins for binding data
        agg1=i,
        agg2=np.nansum)

    sasa_filt[name] = sasa_corr
    flex_filt[name] = flex_corr

    print(f"Filtered lectins - {name}")
    sasa_filt_abs_median = float(sasa_corr.abs().median())
    print(f"sasa_filt_abs_median : {sasa_filt_abs_median:.2f}")
    flex_filt_abs_median = float(flex_corr.abs().median())
    print(f"flex_filt_abs_median: {flex_filt_abs_median:.2f}")
    print("")


"""
All Lectins
agg1 = fixed (nansum)
agg2 = variable (sum/mean/max)
"""
sasa_all = {}
flex_all = {}

for i, name in zip(agg_list, agg_n):
    sasa_corr, flex_corr = get_correlations5(
        all_valid_lectins,  # Use all valid lectins
        glycan_dict,
        binding_data.loc[x, all_valid_lectins],  # Use same lectins for binding data
        agg1=np.nansum,
        agg2=i)

    sasa_all[name] = sasa_corr
    flex_all[name] = flex_corr

    print(f"All lectins - {name}")
    sasa_all_abs_median = float(sasa_corr.abs().median())
    print(f"sasa_all_abs_median: {sasa_all_abs_median:.2f}")
    flex_all_abs_median = float(flex_corr.abs().median())
    print(f"flex_all_abs_median: {flex_all_abs_median:.2f}")
    print("")
