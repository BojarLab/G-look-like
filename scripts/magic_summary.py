from inspect import unwrap
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.graph import glycan_to_nxGraph
import numpy as np
import pandas as pd
from scripts.load_data import load_data


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
        "motif": ["Gal(b1-3)GalNAc", "GalOS(b1-3)GalNAc"], #OS??
        "termini_list": [["f", "f"], ["f", "f"]]
    },




    "AIA": {
        "motif": ["Gal(b1-3)GalNAc", "GlcNAC(b1-3)GalNAc"],
        "termini_list": [["f"], ["f"]]
    },
    "CF": {
        "motif": ["GalNAc", "GalNAc",  "GlcNAc", "GlcNAc"], #repeat is not needed
        "termini_list": [["t", "f", "t", "f"]]
    },
    "HAA": {
        "motif": ["GalNAc", "GlcNAc"], # linkage is missing
        "termini_list": [["t", "t"]]
    },
    "MPA": {
        "motif": ["Gal(b1-3)GlcNAc", "GlcNAc(b1-3)GalNAc"],
        "termini_list": [["f", "f"], ["t", "t"]]
    },
    "LAA": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc"], #why only fuc is terminal and the  ain brach is flexible?
        "termini_list": [["t", "t", "t"]] #t,f,f
    },
    "LcH": {
        "motif": ["Man(a1-6)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc", "Man(a1-2)Man"],
        "termini_list": [["f", "f", "f", "f", "t", "f"], ["t", "t"]]
    },
    "LTL": {
        "motif": ["Gal(b1-4)[Fuc(a1-3)]GlcNAc", "Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]GlcNAc]"], #can we put fuc or gal in branch?
        "termini_list": [["t", "t", "t"], ["t", "t", "t", "t"]]
    },
    "LTA": {
        "motif": ["Gal(b1-4)[Fuc(a1-3)]GlcNAc", "Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]GlcNAc]"],
        "termini_list": [["t", "t", "t"], ["t", "t", "t", "t"]]
    },
    "PSA": {
        "motif": ["Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"],
        "termini_list": [["f", "f", "f", "f", "t","f"]]
    },



    "PTL-I": {
        "motif": ["GalNAc(a1-3)Gal(a1-2)Fuc", "Gal(a1-3)Gal(a1-2)Fuc"], #why not just one main branch?
        "termini_list": [["t", "t", "t"], ["t", "t", "t"]]
    },
    "PTA-I": {
        "motif": ["GalNAc(a1-3)Gal(a1-2)Fuc", "Gal(a1-3)Gal(a1-2)Fuc"],
        "termini_list": [["t", "t", "t"], ["t", "t", "t"]]
    },


    "PTA-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ], #basically only Fuc is temrinal
        "termini_list": [["t", "t", "t"] ] #PTA has only one binding motif so why flexible?
    },
    "PTL-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ],
        "termini_list": [["t", "t", "t"] ] #same here why flexible?
    },


    "TJA-II": {
        "motif": ["Fuc(a1-2)Gal(b1-3)GalNAc", "Fuc(a1-2)Gal(b1-3/4)GlcNAc"],
        "termini_list": [["t", "t", "t"], ["t", "t", "t"]]
    },
    "UEA-I": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc", "Fuc(a1-2)Gal(b1-4)Glc","Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]GlcNAc" ],
        "termini_list": [["t", "t", "t"],["t", "t", "t"],["t", "t", "t"]]
    },
    "CTB": {
        "motif": ["Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)", "Fuc(a1-2)Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)"],
        "termini_list": [["t", "t", "t", "t", "t"], ["t", "t", "t", "t", "t", "t"]]

    },

    "MAL-I": {
        "motif": ["3SGal(b1-4)GlcNAc", "Sia(a2-3)Gal(b1-4)GlcNAc"], #3S, +-6S
        "termini_list": [["t", "t"], ["t", "t", "t"]]
    },
    "MAA": {
        "motif": ["3SGal(b1-4)GlcNAc", "Sia(a2-3)Gal(b1-4)GlcNAc"],  # 3S, +-6S
        "termini_list": [["t", "t"], ["t", "t", "t"]]
    },
    "MAL": {
        "motif": ["3SGal(b1-4)GlcNAc", "Sia(a2-3)Gal(b1-4)GlcNAc"],  # 3S, +-6S
        "termini_list": [["t", "t"], ["t", "t", "t"]]
    },


    "PSL": {
        "motif": ["Sia(a2-6)Gal(b1-3/4)GlcNAc"],
        "termini_list": [["t", "t", "t"]]
    },
    "TJA-I": {
        "motif": ["Sia(a2-6)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "t", "t"]]
    },



    "GS-II": {
        "motif": ["GlcNAc(b1-?)", "GlcNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },
    "PWA": {
        "motif": ["GlcNAc(b1-4)GlcNAc(b1-4)GlcNAc(b1-4)GlcNAc(b1-4)"],
        "termini_list": [["t", "t", "t", "t"]]
    },
    "UEA-II": {
        "motif": ["GlcNAc(b1-3)Gal", "GlcNAc(b1-3)GalNAc", "Fuc(a1-2)Gal(b1-4)GlCNAc", "Fuc(a1-2)Gal(b1-3)GalNAc(b1-3)" ],
        "termini_list": [["t", "t"], ["t", "t"], ["t", "t", "t"], ["t", "t", "t"]]
    },
    "WGA": {
        "motif": ["GlcNAc(b1-?)", "GlcNAc(a1-?)", "GalNAc(a/b1-?)", "Sia(a2-?)", "MurNAc(b1-?)"], #b1? or b2?
        "termini_list": [["t"], ["t"], ["t"], ["t"], ["t"]]
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
        "termini_list": [["t", "t"], ["t", "t"]]
    },
    "GS-I": {
        "motif": ["Gal(a1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },

    "LEA": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },
    "LEL": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },

    "MOA": {
        "motif": ["Gal(a1-3)Gal", "Gal(a1-3)GalNAc"], #Fuc is trimmed
        "termini_list": [["t", "t"], ["t", "f"]] #t,f or t,t,?
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
        "motif": ["Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)", "Gal(a1-3)[Fuc(a1-2)]Gal", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t", "t", "t"], ["t", "t", "t"], ["t", "t"]]
    },

    "STA": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },
    "STL": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },

    "CSA": {
        "motif": ["GalNAc(b1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },
    "DBA": {
        "motif": ["GalNAc(a1-3)GalNAc(b1-?)"],
        "termini_list": [["t", "t"]]
    },
    "SBA": {
        "motif": ["GalNAc(b1-?)", "GalNAc(a1-3)GalNAc(b1-?)", "GalNAc(a1-?"],
        "termini_list": [["t"], ["t", "t"], ["t"]]
    },


    "VVL": {
        "motif": ["GalNAc(b1-3/4)GlcNAc", "GalNAc(b1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t", "f"], ["t"], ["t"]]
    },
    "VVA": {
        "motif": ["GalNAc(b1-3/4)GlcNAc", "GalNAc(b1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t", "f"], ["t"], ["t"]]
    },


    "WFA": {
        "motif": ["GalNAc(b1-?)", "GalNAc(a1-?)", "GalNAc(b1-3/4)GlcNAc(b1-3)"],
        "termini_list": [["t"], ["t"], ["t", "t"]]
    }
}
structure_graphs, glycan_binding, invalid_graphs = load_data()

#lectins = {k:v for k,v in lectin_binding_motif.items() if any(len(t)>1 for t in v['termini_list']) and k in glycan_binding.protein.tolist()}
lectins = {k:v for k,v in lectin_binding_motif.items() if k in glycan_binding.protein.tolist()}


binding_data = glycan_binding.set_index('protein').drop(['target'], axis=1).T
#glycan_dict = {g:v for g,v in structure_graphs.items() if g in binding_data.index or any(compare_glycans(g, b) for b in binding_data.index)}
glycan_dict = {g:v for g,v in structure_graphs.items() if g in binding_data.index and any(compare_glycans(g, b) for b in binding_data.index)}


agg1= np.nansum
agg2= np.nanmean

def get_correlations(lectins, glycan_dict, binding_data, agg1=np.nansum, agg2=np.mean):
  sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())
  for lectin, binding_motif in lectins.items():
    motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided', termini_list=binding_motif['termini_list'][i]) for i in range(len(binding_motif['motif']))]
    all_sasa, all_flex = [], []
    for _, ggraph in glycan_dict.items():
      _, matches = zip(*[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])
      all_sasa.append(agg2([agg1([ggraph.nodes()[n].get('SASA', np.nan) for n in m]) for m in unwrap(matches[0])]))
      all_flex.append(agg2([agg1([ggraph.nodes()[n].get('flexibility', np.nan) for n in m]) for m in unwrap(matches[0])]))
    sasa_df[lectin] = all_sasa
    flex_df[lectin] = all_flex
  sasa_corr = sasa_df.corrwith(binding_data)
  flex_corr = flex_df.corrwith(binding_data)
  return sasa_corr, flex_corr

sasa_corr, flex_corr = get_correlations(lectins, glycan_dict, binding_data.loc[list(glycan_dict.keys()), list(lectins.keys())])







def get_correlations_(lectins, glycan_dict, binding_data, agg1, agg2):
  sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())
  for lectin, binding_motif in lectins.items():
    motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided', termini_list=binding_motif['termini_list'][i]) for i in range(len(binding_motif['motif']))]
    all_sasa, all_flex = [], []
    for _, ggraph in glycan_dict.items():
      _, matches = zip(*[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])

      unwrapped_matches = unwrap(matches[0])

      if not unwrapped_matches:
          all_sasa.append(np.nan)  # No matches found, use NaN
          all_flex.append(np.nan)
      else:
          sasa_values = [agg1([ggraph.nodes[n].get('SASA', np.nan) for n in m])
                         for m in unwrapped_matches]
          flex_values = [agg1([ggraph.nodes[n].get('flexibility', np.nan) for n in m])
                         for m in unwrapped_matches]

          all_sasa.append(agg2(sasa_values) if sasa_values else np.nan)
          all_flex.append(agg2(flex_values) if flex_values else np.nan)

    sasa_df[lectin] = all_sasa
    flex_df[lectin] = all_flex

    # Calculate correlations and process them
    sasa_corr = sasa_df.corrwith(binding_data, axis=0, drop=False, method='pearson') #default
    flex_corr = flex_df.corrwith(binding_data, axis=0, drop=False, method='pearson')

    # Get absolute sums (a single metric for overall correlation strength)
    sasa_abs_medain = sasa_corr.abs().median()
    flex_abs_medain = flex_corr.abs().median()

    # Name and save the results
    sasa_corr.name = "SASA_corr"
    flex_corr.name = "FLEX_corr"
    sasa_corr.to_csv(f'results/stats/sasa_{agg1.__name__}_{agg2.__name__}.csv', header=True)
    flex_corr.to_csv(f'results/stats/flex_{agg1.__name__}_{agg2.__name__}.csv', header=True)

    # Create a simple summary dataframe
    summary = pd.DataFrame({
        'metric': ['SASA', 'Flexibility'],
        'abs_corr_sum': [sasa_abs_medain, flex_abs_medain]
    })
    summary.to_csv(f'results/stats/summary_{agg1.__name__}_{agg2.__name__}.csv', header=True)

  return sasa_corr, flex_corr
#sasa, flex = get_correlations(lectins, glycan_dict, binding_data.loc[list(glycan_dict.keys()), list(lectins.keys())], agg1, agg2)

def compare_aggregations(lectins, glycan_dict, binding_data):
    """
    Compares different combinations of aggregation functions for analyzing SASA and flexibility
    correlations with binding data, with sorted summary statistics.

    Parameters:
    -----------
    lectins : dict
        Dictionary containing lectin binding motif information
    glycan_dict : dict
        Dictionary mapping glycan IDs to their graph representations
    binding_data : pandas.DataFrame
        Binding affinity data for each glycan-lectin pair

    Returns:
    --------
    sasa_comparison : pandas.DataFrame
        DataFrame comparing SASA correlation metrics for different aggregation strategies
    flex_comparison : pandas.DataFrame
        DataFrame comparing flexibility correlation metrics for different aggregation strategies
    summary_df : pandas.DataFrame
        DataFrame with comparison metrics for different aggregation strategies
    """
    # Define the aggregation function combinations to test
    agg_combinations = [
        (np.nansum, np.nanmax, "sum_max"),
        (np.nansum, np.nanmean, "sum_mean"),
        (np.nansum, np.nansum, "sum_sum"),
        (np.nanmean, np.nanmax, "mean_max"),
        (np.nanmean, np.nanmean, "mean_mean"),
        (np.nanmean, np.nansum, "mean_sum")
    ]

    # Initialize result storage
    sasa_results = {}
    flex_results = {}
    summary_rows = []

    # Run correlations with each aggregation combination
    for agg1, agg2, combo_name in agg_combinations:

        sasa_corr, flex_corr = get_correlations(
            lectins,
            glycan_dict,
            binding_data.loc[list(glycan_dict.keys()), list(lectins.keys())],
            agg1,
            agg2
        )

        sasa_results[combo_name] = sasa_corr
        flex_results[combo_name] = flex_corr

        # Calculate only the median of absolute correlations
        sasa_abs_median = sasa_corr.abs().sum()
        flex_abs_median = flex_corr.abs().sum()

        # Store only the median statistics
        summary_rows.append({
            'agg_combo': combo_name,
            'agg1': agg1.__name__,
            'agg2': agg2.__name__,
            'sasa_abs_median': sasa_abs_median,
            'flex_abs_median': flex_abs_median,
            'combined_abs_median': (sasa_abs_median + flex_abs_median) / 2
        })

    # Create comparison DataFrames
    sasa_comparison = pd.DataFrame(sasa_results)
    flex_comparison = pd.DataFrame(flex_results)

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Save comparison results
    sasa_comparison.to_csv('results/stats/sasa_comparison.csv')
    flex_comparison.to_csv('results/stats/flex_comparison.csv')
    summary_df.to_csv('results/stats/aggregation_summary.csv', index=False)

    # Print full summary sorted by combined median (descending)
    sorted_summary = summary_df.sort_values(by='combined_abs_median', ascending=False)
    print("\nSummary of Aggregation Strategies (Median of Absolute Correlations):")
    print(sorted_summary[['agg_combo', 'sasa_abs_median', 'flex_abs_median', 'combined_abs_median']])

    return sasa_comparison, flex_comparison, summary_df
#sasa_comparison, flex_comparison, summary_df  = compare_aggregations(lectins, glycan_dict, binding_data.loc[list(glycan_dict.keys()), list(lectins.keys())])