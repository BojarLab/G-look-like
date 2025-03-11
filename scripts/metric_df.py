import pandas as pd
from scripts.load_data import load_data
from scripts.filter_bind_df import filter_bind_df
from scripts.get_glycan_dict import get_glycan_dict
from scripts.glycan_metric_dict import glycan_metric_dict

def metric_df(lectin, properties):
    """
    Generate a metrics DataFrame for a given lectin and its properties.
    """
    flex_data, binding_df, invalid_graphs =load_data()
    filtered_df = filter_bind_df(binding_df, lectin)
    if filtered_df.empty:
        print(f"No binding data found for {lectin}.")
        return pd.DataFrame()

    glycan_scores = get_glycan_dict(filtered_df)
    metric_dict= glycan_metric_dict(properties, glycan_scores, flex_data)

    metric_df = pd.DataFrame(metric_dict)
    if 'glycan' in metric_df.columns:
        metric_df.set_index('glycan', inplace=True)
    else:
        print(f"⚠️ Warning: 'glycan' column missing for {lectin}. Skipping index setting.")

    metric_df.to_excel(f'results/metric_df/{lectin}_metrics.xlsx', index=True, header=True)
    return metric_df
