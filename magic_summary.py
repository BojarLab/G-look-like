from inspect import unwrap
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.graph import glycan_to_nxGraph
from scripts_dep.load_data import load_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



lectin_binding_motif = {
    "AOL": { "motif": ["Fuc(a1-?)"],
             "termini_list": [["t"]] },
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
        "motif": ["GlcNAc(b1-2/4)Man(a1-?)","Man(a1-2)"],
        "termini_list": [["t","f"], ["t"]]
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
        "termini_list": [["f", "f","f","f", "f"]]
    },
    "CA": {
        "motif": ["Gal(b1-3/4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-3/4)GlcNAc(b1-2)Man(a1-6)]Man"],
        "termini_list": [["f", "f","f","f", "f", "f", "f"]]
    },
    "CAA": {
        "motif": ["Gal(b1-3/4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-3/4)GlcNAc(b1-2)Man(a1-6)]Man"],
        "termini_list": [["f", "f","f","f", "f", "f", "f"]]
    },
    "TL": {
        "motif": ["GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"],
        "termini_list": [["f", "f","f","f", "f", "f", "f", "f"]]
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
        "motif": ["Fuc(a1-3)[Gal(b1-4)]GlcNAc",],
        "termini_list": [["t", "f", "f"]]
    },
    "LTA": {
        "motif": ["Fuc(a1-3)[Gal(b1-4)]GlcNAc",],
        "termini_list": [["t", "f", "f"]]
    },
    "PSA": {
        "motif": ["Fuc(a1-6)"],
        "termini_list": [["t"]]
    },



    "PTL-I": {
        "motif": ["Fuc(a1-2)[GalNAc(a1-3)]Gal", "Fuc(a1-2)[Gal(a1-3)]Gal"],
        "termini_list": [["t", "t", "f"], ["t", "t", "f"]]
    },
    "PTA-I": {
        "motif": ["Fuc(a1-2)[GalNAc(a1-3)]Gal", "Fuc(a1-2)[Gal(a1-3)]Gal"],
        "termini_list": [["t", "t", "f"], ["t", "t", "f"]]
    },


    "PTA-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ],
        "termini_list": [["t", "f", "f"] ]
    },
    "PTL-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ],
        "termini_list": [["t", "f", "f"] ]
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
        "motif": ["Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)", "Fuc(a1-2)Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)"],
        "termini_list": [["t", "f", "t", "f", "f"], ["t", "f", "f", "t", "f", "f"]]

    },

    "MAL-I": {
        "motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },
    "MAA": {
        "motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },
    "MAL": {
        "motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
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
        "motif": ["GlcNAc(b1-3)", "Fuc(a1-2)Gal(b1-4)GlcNAc", "Fuc(a1-2)Gal(b1-3)GalNAc" ],
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
structure_graphs, glycan_binding, invalid_graphs = load_data()
lectins_filt = {k:v for k,v in lectin_binding_motif.items() if any(len(t)>1 for t in v['termini_list']) and k in glycan_binding.protein.tolist()} # compare_aggregations
lectins = {k:v for k,v in lectin_binding_motif.items() if k in glycan_binding.protein.tolist()}
binding_data = glycan_binding.set_index('protein').drop(['target'], axis=1).T
glycan_dict = {g:v for g,v in structure_graphs.items() if g in binding_data.index or any(compare_glycans(g, b) for b in binding_data.index)}



def plot_correlation_boxplot(sasa_all, flex_all, title="Correlation Distributions"):
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

def get_correlations_(lectins, glycan_dict, binding_data, agg1, agg2):
    sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())

    def safe_agg(func, data):
        if len(data) == 0:
            return np.nan  # Return NaN for empty arrays
        return func(data)

    for lectin, binding_motif in lectins.items():
        motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided',
                                          termini_list=binding_motif['termini_list'][i]) for i in
                        range(len(binding_motif['motif']))]
        all_sasa, all_flex = [], []
        for _, ggraph in glycan_dict.items():
            _, matches = zip(
                *[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])

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
    sasa_corr = sasa_corr.reset_index().groupby(
        "index").median().dropna()  # agg multiple proteins columns in binding data

    flex_corr = flex_df.corrwith(binding_data).dropna()  # drop lectins with no corr
    flex_corr = flex_corr.reset_index().groupby("index").median()
    return sasa_corr, flex_corr

"""
Lectin-Binding Motif > 1 mono
agg1 = variable
agg2 = fixed

"""
agg_list = [np.nansum, np.nanmax, np.nanmean]
agg_n = ['nansum', 'nanmax', 'nanmean']


sasa_filt = {}
flex_filt = {}
for i, name in zip(agg_list, agg_n):
    sasa_corr, flex_corr = get_correlations_(
        lectins_filt,
        glycan_dict,
        binding_data.loc[list(glycan_dict.keys()), list(lectins_filt.keys())],
        agg1=i,
        agg2=np.nansum)

    sasa_filt[name] = sasa_corr
    flex_filt[name] = flex_corr

    sasa_filt_abs_median = sasa_corr.abs().median()
    print(f"sasa_filt_abs_median : {sasa_filt_abs_median[0]:.2f}")
    flex_filt_abs_median = flex_corr.abs().median()
    print(f"flex_filt_abs_median: {flex_filt_abs_median[0]:.2f}")
    print("")


plot_correlation_boxplot(sasa_filt, flex_filt, "Correlation Distribution Filtered Lectins")
plt.savefig("results/plots/correlation_boxplot_filtered_lectins.png", dpi=300)
plt.show()


"""
all Lectins
agg1 = fixed
agg2 = variable
"""

sasa_all = {}
flex_all = {}
for i, name in zip(agg_list, agg_n):
    sasa_corr, flex_corr = get_correlations_(
        lectins,
        glycan_dict,
        binding_data.loc[list(glycan_dict.keys()), list(lectins.keys())],
        agg1=np.nansum,
        agg2=i)

    sasa_all[name] = sasa_corr
    flex_all[name] = flex_corr

    sasa_all_abs_median = sasa_corr.abs().median()
    print(f"sasa_all_abs_median: {sasa_all_abs_median[0]:.2f}")
    flex_all_abs_median = flex_corr.abs().median()
    print(f"flex_all_abs_median: {flex_all_abs_median[0]:.2f}")
    print("")

plot_correlation_boxplot(sasa_all, flex_all, "Correlation Distribution all Lectins")
plt.savefig("results/plots/correlation_boxplot_all_lectins.png", dpi=300)
plt.show()







