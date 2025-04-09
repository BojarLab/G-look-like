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

count = 0
for lectin_entry in cosm["Lectin Name"]:
    try:
        names = ast.literal_eval(lectin_entry)
    except:
        names = [lectin_entry]


    for name in names:
        if (name not in oracle.name.values and
                name not in more.name.values and
                name not in lectin_specificity.name.values
                and name not in glycan_binding.protein.values):
            print(name)
            count += 1