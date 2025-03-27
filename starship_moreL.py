from glycowork.motif.graph import compare_glycans, subgraph_isomorphism
from glycowork.motif.graph import glycan_to_nxGraph
from scripts.load_data import load_data_v4
import pandas as pd
import numpy as np
from glycowork.glycan_data.loader import unwrap
from scipy import stats
from glycowork.motif.processing import get_class
import seaborn as sns
import matplotlib.pyplot as plt
from glycowork.glycan_data.loader import lectin_specificity






lectin_binding_motif = {}
lectin_keys = {}

for index, row in lectin_specificity[["abbreviation", "specificity_primary"]].iterrows():
    keys = [l.strip() for l in row["abbreviation"].split(",")]
    main_key = keys[-1]  # Use the last abbreviation as the main key
    for key in keys:
        lectin_keys[key] = main_key
    specificity_data = row["specificity_primary"]
    motif_list = list(specificity_data.keys())
    termini_list = list(specificity_data.values())
    lectin_binding_motif[main_key] = {
        "motif": motif_list,
        "termini_list": termini_list}


structure_graphs, glycan_binding, invalid_graphs = load_data_v4()
binding_data = glycan_binding.set_index('protein').drop(['target'], axis=1)
idx = [k for k in binding_data.index if k in lectin_keys]
binding_data = binding_data.loc[idx,:]
binding_data.index = [lectin_keys[k] for k in binding_data.index]
binding_data = binding_data.groupby(level=0).median().dropna(axis=1, how='all').T

glycan_dict = {g:v for g,v in structure_graphs.items() if g in binding_data.index or any(compare_glycans(g, b) for b in binding_data.index)}
lectins = {v for k,v in lectin_keys.items() if k in lectin_binding_motif and k in glycan_binding.protein.tolist()}
binding_data_filt = binding_data.loc[list(glycan_dict.keys()), list(lectins)]
print(binding_data_filt.shape)


def plot_correlation(lectin_name, binding_data, filepath=''):
  p_value_f = 1.0
  p_value_s = 1.0
  sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())
  binding_motif = lectin_binding_motif[lectin_name]


  motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided', termini_list=binding_motif['termini_list'][i]) for i in range(len(binding_motif['motif']))]
  all_sasa, all_flex, motif_indices = [], [], []
  for glycan, ggraph in glycan_dict.items():
    _, matches = zip(*[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])
    motif_idx = next((i for i, iso in enumerate(matches) if iso), None)
    motif_indices.append(motif_idx)
    matches_unwrapped = unwrap(matches)
    all_sasa.append(np.nanmean([np.nanmean([ggraph.nodes()[n].get('SASA', np.nan) for n in m]) for m in matches_unwrapped]) if matches_unwrapped else np.nan)
    all_flex.append(np.nanmean([np.nanmean([ggraph.nodes()[n].get('flexibility', np.nan) for n in m]) for m in matches_unwrapped]) if matches_unwrapped else np.nan)

  sasa_df['SASA'] = all_sasa
  flex_df['flexibility'] = all_flex
  sasa_df['binding'] = binding_data[lectin_name]
  flex_df['binding'] = binding_data[lectin_name]
  sasa_df['motif'] = [binding_motif['motif'][i] if i is not None else 'None' for i in motif_indices]
  flex_df['motif'] = [binding_motif['motif'][i] if i is not None else 'None' for i in motif_indices]

  # Create the figure and subplots
  classes = [get_class(g) if get_class(g) else 'other' for g in glycan_dict]
  sasa_df['class'] = classes
  flex_df['class'] = classes
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
  # Plot SASA correlation with seaborn
  clean_sasa = sasa_df.dropna()
  sns.scatterplot(x='SASA', y='binding', hue='class', style='motif', data=clean_sasa, ax=ax1,
  hue_order = ['N', 'O', "lipid/free", ""],
  palette = "tab10")

  # Regression with confidence interval using seaborn
  if len(clean_sasa) > 1:
    sns.regplot(x='SASA', y='binding', data=clean_sasa, ax=ax1, scatter=False, color='r', ci=95)
    slope, intercept, r_value, p_value_s, std_err = stats.linregress(clean_sasa['SASA'], clean_sasa['binding'])
    ax1.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}\nr = {r_value:.3f}, p = {p_value_s:.3f}',
           transform=ax1.transAxes, verticalalignment='top')
  ax1.set_title(f'SASA vs Binding for {lectin_name}')
  ax1.set_xlabel('SASA')
  ax1.set_ylabel('Binding')
  legend = ax1.get_legend()
  if legend is not None:
      legend.remove() # Remove legend from first plot
  # Plot flexibility correlation with seaborn
  clean_flex = flex_df.dropna()
  sns.scatterplot(x='flexibility', y='binding', hue='class', style='motif', data=clean_flex, ax=ax2,
        hue_order=['N', 'O', "lipid/free", ""],
        palette="tab10")
  # Regression with confidence interval using seaborn
  if len(clean_flex) > 1:
    sns.regplot(x='flexibility', y='binding', data=clean_flex, ax=ax2, scatter=False, color='r', ci=95)
    slope, intercept, r_value, p_value_f, std_err = stats.linregress(clean_flex['flexibility'], clean_flex['binding'])
    ax2.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}\nr = {r_value:.3f}, p = {p_value_f:.3f}',
           transform=ax2.transAxes, verticalalignment='top')
  ax2.set_title(f'Flexibility vs Binding for {lectin_name}')
  ax2.set_xlabel('Flexibility')
  ax2.set_ylabel('Binding')
  # Move legend outside the plot
  handles, labels = ax2.get_legend_handles_labels()
  ax2.legend(handles, labels, title='Class & Motif', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout()

  # Save figure if filepath is provided
  if filepath and p_value_s < 0.05 or p_value_f < 0.05:
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
  return fig, ax1, ax2

for k, v in lectin_keys.items():
    try:
        plot_correlation(v, binding_data_filt, f'results/plots/{v}_correlation.pdf')
    except Exception as e:
        print(f"Error occurred: {e}")

