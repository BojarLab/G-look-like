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
from adjustText import adjust_text
import warnings
warnings.filterwarnings("ignore")

lectin_binding_motif = {}
lectin_keys = {}


"""# Load the lectin specificity data
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
        "termini_list": termini_list}"""



"""
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


"""


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

"""for k, v in lectin_keys.items():
    try:
        plot_correlation(v, binding_data_filt, f'results/plots/dev/{v}_correlation.pdf')
    except Exception as e:
        print(f"Error occurred: {e}")"""

"""for row in lectin_specificity["abbreviation"]:
    for l in row.split(","):
        l = l.strip()
        if l not in lectin_binding_motif.keys():
            print(l)
            len(l)
            continue
            """

structure_graphs, glycan_binding, invalid_graphs = load_data_v4()
cosm = pd.read_csv('data/glycosmos_lectins/download.csv')

# Initialize dictionaries
lectin_keys = {}
lectin_binding_motif = {}

# First, process Glycosmos data
for index, row in cosm.iterrows():
    if pd.isna(row["Monosaccharide Specificities"]):
        continue

    import ast

    try:
        synonyms = ast.literal_eval(row["Lectin Name"])
    except:
        synonyms = [s.strip(' "\'[]') for s in row["Lectin Name"].split(',')]

    synonyms = [s for s in synonyms if s]

    if len(synonyms) >= 2:
        canonical_name = synonyms[1]
    else:
        canonical_name = synonyms[0] if synonyms else "Unknown"

    # Add to lectin_keys
    for synonym in synonyms:
        clean_synonym = synonym.strip(' "\'[]')
        lectin_keys[clean_synonym] = canonical_name

    # Parse monosaccharide specificities for lectin_binding_motif
    try:
        specificity_data = ast.literal_eval(row["Monosaccharide Specificities"])

        if isinstance(specificity_data, dict):
            motif_list = list(specificity_data.keys())
            termini_list = list(specificity_data.values())
        else:
            motif_list = [specificity_data] if not isinstance(specificity_data, list) else specificity_data
            termini_list = [["t"]] * len(motif_list)
    except:
        motif_list = [str(row["Monosaccharide Specificities"]).strip()]
        termini_list = [["t"]]

    # Add to lectin_binding_motif using the canonical name
    lectin_binding_motif[canonical_name] = {
        "motif": motif_list,
        "termini_list": termini_list
    }

# Now add all lectins from binding_data to lectin_keys if not already present
all_binding_lectins = glycan_binding['protein'].unique()
for lectin in all_binding_lectins:
    if lectin not in lectin_keys:
        # If not in lectin_keys, add it with itself as the canonical name
        lectin_keys[lectin] = lectin

        # Also add a default motif if it doesn't have one
        if lectin not in lectin_binding_motif:
            # Try to find this lectin in cosm dataframe
            lectin_rows = cosm[cosm["Lectin Name"].str.contains(lectin, na=False)]

            if not lectin_rows.empty:
                # Use the specificity from the first matching row
                row = lectin_rows.iloc[0]

                try:
                    specificity_data = ast.literal_eval(row["Monosaccharide Specificities"])

                    if isinstance(specificity_data, dict):
                        motif_list = list(specificity_data.keys())
                        termini_list = list(specificity_data.values())
                    else:
                        motif_list = [specificity_data] if not isinstance(specificity_data, list) else specificity_data
                        termini_list = [["t"]] * len(motif_list)
                except:
                    # If parsing fails, use the raw string as a single motif
                    motif_str = str(row["Monosaccharide Specificities"]).strip()
                    motif_list = [motif_str] if motif_str and not pd.isna(motif_str) else ["Unknown"]
                    termini_list = [["t"]]
            else:
                # If no match in cosm, use a default "Unknown" motif
                motif_list = ["Unknown"]
                termini_list = [["t"]]

            lectin_binding_motif[lectin] = {
                "motif": motif_list,
                "termini_list": termini_list
            }

# Filter binding_data to include all lectins, replacing with canonical names
binding_data = glycan_binding.set_index('protein').drop(['target'], axis=1)
found_synonyms = [k for k in binding_data.index if k in lectin_keys]

binding_data = binding_data.loc[found_synonyms, :]
binding_data.index = [lectin_keys[k] for k in binding_data.index]
binding_data = binding_data.groupby(level=0).median().dropna(axis=1, how='all').T

print(f"Total lectins with motifs: {len(lectin_binding_motif)}")
print(f"Total lectin synonyms: {len(lectin_keys)}")
print(f"Lectin synonyms found in binding data: {len(found_synonyms)}")
print(f"Unique lectins in final dataset: {len(binding_data.columns)}")

# Correct the filtering logic for the final dataset
glycan_dict = {g: v for g, v in structure_graphs.items()
               if g in binding_data.index or any(compare_glycans(g, b) for b in binding_data.index)}

# Get all unique canonical lectin names that appear in the binding data AND have motifs
lectins = {col for col in binding_data.columns if col in lectin_binding_motif}

# Filter the binding data to only include relevant glycans and lectins with motifs
binding_data_filt = binding_data.loc[list(glycan_dict.keys()), list(lectins)]

# Print how many lectins were filtered out due to missing motifs
filtered_out = set(binding_data.columns) - lectins
print(f"Lectins filtered out due to missing motifs: {len(filtered_out)}")
if filtered_out:
    print(f"Examples of filtered lectins: {list(filtered_out)[:5]}")
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


def plot_correlation_scatter(out, filepath=''):
  def assign_quadrant(row):
    if row["SASA_corr"] >= 0 and row["flex_corr"] >= 0:
      return "Q1: High SASA, High Flex", "darkgreen"
    elif row["SASA_corr"] < 0 and row["flex_corr"] >= 0:
      return "Q2: Low SASA, High Flex", "royalblue"
    elif row["SASA_corr"] < 0 and row["flex_corr"] < 0:
      return "Q3: Low SASA, Low Flex", "darkred"
    else:
      return "Q4: High SASA, Low Flex", "darkorange"
  out["quadrant"], out["color"] = zip(*out.apply(assign_quadrant, axis=1))
  fig, ax = plt.subplots(figsize=(12, 10))
  scatter = sns.scatterplot(
    x="SASA_corr",
    y="flex_corr",
    data=out,
    s=100,
    hue="quadrant",
    palette=dict(zip(out["quadrant"].unique(), out["color"].unique())),
    ax=ax
  )
  plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
  plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
  plt.title("Correlation between SASA and Flexibility", fontsize=14)
  plt.xlabel("SASA Correlation", fontsize=12)
  plt.ylabel("Flexibility Correlation", fontsize=12)
  plt.grid(True, alpha=0.3)
  texts = []
  for idx, row in out.iterrows():
    texts.append(ax.annotate(
      idx,
      (row["SASA_corr"], row["flex_corr"]),
      fontsize=10,
      fontweight="bold",
      color=row["color"]
    ))
  adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))
  ax.fill_between([-1, 0], 0, 1, alpha=0.1, color="royalblue")
  ax.fill_between([0, 1], 0, 1, alpha=0.1, color="darkgreen")
  ax.fill_between([-1, 0], -1, 0, alpha=0.1, color="darkred")
  ax.fill_between([0, 1], -1, 0, alpha=0.1, color="darkorange")
  ax.set_xlim(-0.8, 1.1)
  ax.set_ylim(-0.8, 1.1)
  plt.legend(title="Quadrants", loc="best", framealpha=0.9)
  corr_coef = np.corrcoef(out["SASA_corr"], out["flex_corr"])[0, 1]
  ax.annotate(
    f"Correlation: {corr_coef:.2f}",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    fontsize=12
  )
  plt.tight_layout()
  if filepath:
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
  return fig, ax


def get_lectin_clusters(binding_data, filepath='', agg1=np.nanmean, agg2=np.nanmean):
  sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())
  for lectin in binding_data.columns:
    binding_motif = lectin_binding_motif[lectin]
    motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided', termini_list=binding_motif['termini_list'][i]) for i in range(len(binding_motif['motif']))]
    all_sasa, all_flex = [], []
    for _, ggraph in glycan_dict.items():
      _, matches = zip(*[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])
      matches_unwrapped = unwrap(matches)
      all_sasa.append(agg2([agg1([ggraph.nodes()[n].get('SASA', np.nan) for n in m]) for m in matches_unwrapped]) if matches_unwrapped else np.nan)
      all_flex.append(agg2([agg1([ggraph.nodes()[n].get('flexibility', np.nan) for n in m]) for m in matches_unwrapped]) if matches_unwrapped else np.nan)
    sasa_df[lectin] = all_sasa
    flex_df[lectin] = all_flex
  sasa_corr = sasa_df.corrwith(binding_data)
  flex_corr = flex_df.corrwith(binding_data)
  out = pd.concat([sasa_corr, flex_corr], axis=1)
  out.columns = ['SASA_corr', 'flex_corr']
  plot_correlation_scatter(out.dropna(), filepath=filepath)

get_lectin_clusters(binding_data_filt, filepath='results/plots/dev/cluster.pdf')


"""for l in lectin_binding_motif:
        if l in  binding_data_filt.columns:
            plot_correlation(l, binding_data_filt, f'results/plots/dev/{l}.pdf')
"""