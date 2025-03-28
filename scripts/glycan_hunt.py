import pandas as pd
from glycowork.motif.processing import canonicalize_iupac


df= pd.read_csv("/Users/xakdze/PycharmProjects/G-look-like/data/glycan_hunt/glycan_data_hpylori.csv", sep=";", header="infer")

glycan_c= []
for i in df["glycan"]:
    print(i)
    print(canonicalize_iupac(i))
    glycan_c.append(canonicalize_iupac(i))
df["glycan"]= glycan_c


# Split DataFrame into 20-row chunks and save as CSVs
for i in range(0, len(df), 20):
    chunk = df.iloc[i:i+20]
    chunk.to_csv(f"/Users/xakdze/PycharmProjects/G-look-like/data/glycan_hunt/chunk_{i//20+1}.csv", sep=",", index=False)
