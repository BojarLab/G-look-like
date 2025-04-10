import pandas as pd
from glycowork.glycan_data.loader import lectin_specificity
from scripts.load_data import load_data_v4

oracle= pd.read_csv('data/all_meta.csv')
print(len(oracle.lectin_id.drop_duplicates()))
print(len(oracle.name.drop_duplicates()))

more = pd.read_csv('data/diversity_arrays_meta.csv')
print(len(more.lectin_id.drop_duplicates()))
print(len(more.name.drop_duplicates()))

cosm= pd.read_csv('data/glycosmos_lectins/download.csv')

structure_graphs, glycan_binding, invalid_graphs = load_data_v4()

import ast
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from itertools import combinations


cosm_lectins = set()
for lectin_entry in cosm["Lectin Name"]:
    try:
        names = ast.literal_eval(lectin_entry)
    except:
        names = [lectin_entry]

    for name in names:
        if name:  # Skip empty strings
            cosm_lectins.add(name)

# Get lectin names from other dataframes
oracle_lectins = set(oracle.name.values)
more_lectins = set(more.name.values)
specificity_lectins = set(lectin_specificity.name.values)
binding_lectins = set(glycan_binding.protein.values)  # Added glycan_binding.protein

# Calculate intersections
all_datasets = oracle_lectins & more_lectins & specificity_lectins & cosm_lectins & binding_lectins
oracle_more_spec_cosm = oracle_lectins & more_lectins & specificity_lectins & cosm_lectins
oracle_more_spec_binding = oracle_lectins & more_lectins & specificity_lectins & binding_lectins
oracle_more_cosm_binding = oracle_lectins & more_lectins & cosm_lectins & binding_lectins
oracle_spec_cosm_binding = oracle_lectins & specificity_lectins & cosm_lectins & binding_lectins
more_spec_cosm_binding = more_lectins & specificity_lectins & cosm_lectins & binding_lectins

# Calculate unique lectins in each dataset
only_oracle = oracle_lectins - (more_lectins | specificity_lectins | cosm_lectins | binding_lectins)
only_more = more_lectins - (oracle_lectins | specificity_lectins | cosm_lectins | binding_lectins)
only_specificity = specificity_lectins - (oracle_lectins | more_lectins | cosm_lectins | binding_lectins)
only_cosm = cosm_lectins - (oracle_lectins | more_lectins | specificity_lectins | binding_lectins)
only_binding = binding_lectins - (oracle_lectins | more_lectins | specificity_lectins | cosm_lectins)

# Print results
print(
    f"Total unique lectins across all datasets: {len(oracle_lectins | more_lectins | specificity_lectins | cosm_lectins | binding_lectins)}")
print(f"Lectins in all five datasets: {len(all_datasets)}")
print("\nFour-way intersections:")
print(f"Oracle, More, Specificity, Cosm: {len(oracle_more_spec_cosm)}")
print(f"Oracle, More, Specificity, Binding: {len(oracle_more_spec_binding)}")
print(f"Oracle, More, Cosm, Binding: {len(oracle_more_cosm_binding)}")
print(f"Oracle, Specificity, Cosm, Binding: {len(oracle_spec_cosm_binding)}")
print(f"More, Specificity, Cosm, Binding: {len(more_spec_cosm_binding)}")

print("\nUnique to each dataset:")
print(f"Only in Oracle: {len(only_oracle)}")
print(f"Only in More: {len(only_more)}")
print(f"Only in Specificity: {len(only_specificity)}")
print(f"Only in Cosm: {len(only_cosm)}")
print(f"Only in Binding: {len(only_binding)}")

# Print some examples
print("\nExamples of lectins in all five datasets:")
print(list(all_datasets)[:5])

print("\nExamples of lectins only in Oracle:")
print(list(only_oracle)[:5])

print("\nExamples of lectins only in More:")
print(list(only_more)[:5])

print("\nExamples of lectins only in Specificity:")
print(list(only_specificity)[:5])

print("\nExamples of lectins only in Cosm:")
print(list(only_cosm)[:5])

print("\nExamples of lectins only in Binding:")
print(list(only_binding)[:5])

# Create a DataFrame to store the results
results = pd.DataFrame({
    'Dataset': ['Oracle', 'More', 'Specificity', 'Cosm', 'Binding', 'All Datasets Intersection',
                'Oracle+More+Specificity+Cosm', 'Oracle+More+Specificity+Binding',
                'Oracle+More+Cosm+Binding', 'Oracle+Specificity+Cosm+Binding', 'More+Specificity+Cosm+Binding',
                'Only Oracle', 'Only More', 'Only Specificity', 'Only Cosm', 'Only Binding'],
    'Count': [len(oracle_lectins), len(more_lectins), len(specificity_lectins), len(cosm_lectins), len(binding_lectins),
              len(all_datasets), len(oracle_more_spec_cosm), len(oracle_more_spec_binding),
              len(oracle_more_cosm_binding), len(oracle_spec_cosm_binding), len(more_spec_cosm_binding),
              len(only_oracle), len(only_more), len(only_specificity), len(only_cosm), len(only_binding)]
})

print("\nSummary Table:")
print(results)
