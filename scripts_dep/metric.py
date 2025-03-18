from scripts_dep.load_data import load_data
from scripts_dep.filter_bind_df import filter_bind_df
from scripts_dep.get_glycan_scores import get_glycan_scores
from scripts_dep.find import find_matching_glycan, magic
from glycowork.motif.processing import get_class
import numpy as np
from typing import Dict, List, Union, Any
from networkx import Graph
import pandas as pd


def metric_df(lectin, properties, within, btw):
    """ Step1"""
    flex_data, binding_df, invalid_graphs = load_data()
    filtered_df = filter_bind_df(binding_df, lectin)
    if filtered_df.empty:
        print(f"No binding data found for {lectin}.")
        return pd.DataFrame()
    glycan_scores: Dict[str, float] = get_glycan_scores(filtered_df)

    """ Step2"""
    metric_df = glycan_metric(properties, glycan_scores, flex_data, within, btw)

    if metric_df is not None:
        # Get function names as strings
        within_str = within.__name__ if hasattr(within, '__name__') else str(within).split()[1]
        btw_str = btw.__name__ if hasattr(btw, '__name__') else str(btw).split()[1]

        # Save with function names in the filename
        metric_df.to_excel(f'results/metric_df/{lectin}_metrics_{within_str}_{btw_str}.xlsx', index=True, header=True)

    return metric_df

def glycan_metric(properties: str,
                                glycan_scores: dict,
                                flex_data: dict[str, Graph], within, btw) -> pd.DataFrame:
    """
    Generate metrics for each glycan by processing them one by one and creating metrics at the end.
    """
    metric_dfs = []  # List to store DataFrames for each glycan
    missing_glycans = []

    for glycan in glycan_scores:
        binding_score = glycan_scores[glycan]
        matched_glycan = find_matching_glycan(flex_data, glycan)

        if not matched_glycan:
            missing_glycans.append(glycan)
            continue

        metric_df = magic(matched_glycan, properties, flex_data, within, btw )

        if metric_df is not None:
            metric_df['binding_score'] = binding_score
            metric_df['class'] = get_class(matched_glycan) or np.nan

            metric_dfs.append(metric_df)

    if missing_glycans:
        print(f"Not-matched glycan in flex data:")

    if not metric_dfs:
        print("No metrics could be generated. Returning empty DataFrame.")
        return pd.DataFrame()

    # Combine all metrics into a single DataFrame
    final_df = pd.concat(metric_dfs, ignore_index=True)
    print(f"Processed {len(final_df)} glycans with metrics.")

    if 'glycan' in final_df.columns:
        final_df.set_index('glycan', inplace=True)
    else:
        print(f"⚠️ Warning: 'glycan' column missing. Skipping index setting.")

    return final_df