from inspect import unwrap
from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.graph import glycan_to_nxGraph
from scripts.load_data import load_data
import pandas as pd
import scipy.stats as stats
import itertools

import warnings
warnings.filterwarnings("ignore")


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
    "LTA": "LTL",
    "LTL": "LTL",
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



def get_correlations5(lectins, glycan_dict, binding_data, agg1, agg2):
    """
    Calculate correlations between SASA/flexibility values and binding data.

    Parameters:
    - lectins: List of lectin names to analyze
    - glycan_dict: Dictionary mapping glycan IDs to graph structures
    - binding_data: DataFrame with binding data for lectins
    - agg1: Function for first-level aggregation (node values within a match)
    - agg2: Function for second-level aggregation (across multiple matches)

    Returns: (sasa_correlation, flexibility_correlation) tuple of DataFrames
    """
    # Initialize DataFrames
    sasa_df = pd.DataFrame(index=glycan_dict.keys())
    flex_df = pd.DataFrame(index=glycan_dict.keys())

    for lectin in lectins:
        # Get binding motif for this lectin
        binding_motif = lectin_binding_motif[lectin]

        # Create graph representations of all motifs
        motif_graphs = [
            glycan_to_nxGraph(
                binding_motif['motif'][i],
                termini='provided',
                termini_list=binding_motif['termini_list'][i]
            ) for i in range(len(binding_motif['motif']))
        ]

        # Store results for each glycan
        sasa_values = {}
        flex_values = {}

        # Process each glycan
        for glycan_key, glycan_graph in glycan_dict.items():
            # Collect all matches from all motifs
            all_matches = []
            for motif_graph in motif_graphs:
                _, matches = subgraph_isomorphism(glycan_graph, motif_graph, return_matches=True)
                all_matches.extend(unwrap(matches))

            if not all_matches:
                continue

            # Calculate values for each match
            match_sasa = []
            match_flex = []

            for match in all_matches:
                # Apply first aggregation to node values
                sasa = agg1([glycan_graph.nodes[n].get('SASA', np.nan) for n in match])
                flex = agg1([glycan_graph.nodes[n].get('flexibility', np.nan) for n in match])

                match_sasa.append(sasa)
                match_flex.append(flex)

            # Apply second aggregation or use single value
            if len(match_sasa) > 1:
                sasa_values[glycan_key] = agg2(match_sasa)
                flex_values[glycan_key] = agg2(match_flex)
            else:
                sasa_values[glycan_key] = match_sasa[0] if match_sasa else np.nan
                flex_values[glycan_key] = match_flex[0] if match_flex else np.nan

        # Add results to DataFrames
        sasa_df[lectin] = pd.Series(sasa_values)
        flex_df[lectin] = pd.Series(flex_values)

    # Calculate correlations and handle duplicate lectins
    sasa_correlation = sasa_df.corrwith(binding_data, axis=0, drop=False, method='pearson')
    sasa_correlation = sasa_correlation.reset_index().groupby("index").median()

    flex_correlation = flex_df.corrwith(binding_data)
    flex_correlation = flex_correlation.reset_index().groupby("index").median()

    return sasa_correlation, flex_correlation

import numpy as np

"""
Common setup for analyses of filtered and all lectins
"""
# Define aggregation functions and their labels
agg_functions = [np.nansum, np.nanmax, np.nanmean]
agg_labels = ['nansum', 'nanmax', 'nanmean']

# Prepare data components
glycan_keys = list(glycan_dict.keys())
unique_lectin_names = list(set(lectin_keys.values()))

# Filter lectins to only those present in binding data
filtered_lectins = [lectin for lectin in lectins_filt.keys() if lectin in binding_data.columns]
all_lectins = [lectin for lectin in lectins.keys() if lectin in binding_data.columns]

"""
Analysis 1: Filtered Lectins - varies first-level aggregation (agg1)
Uses fixed second-level aggregation (np.nansum)
"""
sasa_filtered = {}
flex_filtered = {}

print("=== Filtered Lectins Analysis ===")
for agg_func, agg_name in zip(agg_functions, agg_labels):
    # Calculate correlations with varying first-level aggregation
    sasa_corr, flex_corr = get_correlations5(
        filtered_lectins,
        glycan_dict,
        binding_data.loc[glycan_keys, filtered_lectins],
        agg1=np.nanmean,  # Vary this aggregation
        agg2=np.nansum  # Keep this fixed
    )

    # Store results
    sasa_filtered[agg_name] = sasa_corr
    flex_filtered[agg_name] = flex_corr


    # Print metrics
    print(f"Using {agg_name} for first-level aggregation:")
    print(f"  SASA correlation: {float(sasa_corr.abs().median()):.2f}")
    print(f"  Flexibility correlation: {float(flex_corr.abs().median()):.2f}")

"""
Analysis 2: All Lectins - uses fixed first-level aggregation (np.nansum)
Varies second-level aggregation (agg2)
"""
sasa_all = {}
flex_all = {}

print("\n=== All Lectins Analysis ===")
for agg_func, agg_name in zip(agg_functions, agg_labels):
    # Calculate correlations with varying second-level aggregation
    sasa_corr, flex_corr = get_correlations5(
        all_lectins,
        glycan_dict,
        binding_data.loc[glycan_keys, all_lectins],
        agg1=np.nanmean,
        agg2=agg_func  # Vary this aggregation
    )

    # Store results
    sasa_all[agg_name] = sasa_corr
    flex_all[agg_name] = flex_corr

    # Print metrics
    print(f"Using {agg_name} for second-level aggregation:")
    print(f"  SASA correlation: {float(sasa_corr.abs().median()):.2f}")
    print(f"  Flexibility correlation: {float(flex_corr.abs().median()):.2f}")


def perform_t_test(data_dict):
    keys = list(data_dict.keys())  # Get the keys (aggregation labels)
    combinations = list(itertools.combinations(keys, 2))  # Generate combinations of keys

    for comb in combinations:
        key1, key2 = comb

        # Get the DataFrames for each key
        df1, df2 = data_dict[key1], data_dict[key2]

        # Convert to numeric values correctly
        # Extract the first column if it's a DataFrame before applying to_numeric
        df1_clean = pd.to_numeric(df1.iloc[:, 0].dropna(), errors='coerce').values
        df2_clean = pd.to_numeric(df2.iloc[:, 0].dropna(), errors='coerce').values


        # Remove any rows where either array has NaN
        mask = ~np.isnan(df1_clean) & ~np.isnan(df2_clean)
        df1_clean = df1_clean[mask]
        df2_clean = df2_clean[mask]

        print(f"  Final number of paired values: {len(df1_clean)}")

        # Check if there's enough data to perform the t-test
        if len(df1_clean) > 1 and np.nanstd(df1_clean) > 0 and np.nanstd(df2_clean) > 0:
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(df1_clean, df2_clean)

            print(f"T-test for {key1} vs {key2}:")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.3f}")
        else:
            print(f"Cannot perform t-test: insufficient data or zero variance")
    else:
        print(f"Not enough valid data for {key1} vs {key2}")
    print()

# Call the function to perform t-tests
perform_t_test(sasa_filtered) # sum , mean
#perform_t_test(flex_filtered)
