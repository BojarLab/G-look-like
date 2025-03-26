import pickle
import importlib.resources as pkg_resources
import glycontact
import pandas as pd
import networkx as nx


BINDING_DATA_PATH_v3 = '/Users/xakdze/PycharmProjects/G-look-like/data/20250216_glycan_binding.csv'
BINDING_DATA_PATH_v2 = '/Users/xakdze/PycharmProjects/G-look-like/data/20241206_glycan_binding.csv'  # seq,protein

BINDING_DATA_PATH_v4 = "/Users/xakdze/PycharmProjects/G-look-like/data/20250314_glycan_binding_processed.csv"


def load_data_pdb():
    """Load glycan flexibility data from PDB source."""
    with pkg_resources.files(glycontact).joinpath("glycan_graphs.pkl").open("rb") as f:
        return pickle.load(f)

def load_data_v3():
    """Load glycan flexibility and binding data, process graphs, and return results."""
    flex_data = load_data_pdb()
    invalid_graphs = [glycan for glycan in flex_data if not isinstance(flex_data[glycan], nx.Graph)]

    def map_protein_to_target(df_target, df_map):
        """
        Maps protein names to their corresponding targets (sequences) in df_target
        using mapping from df_map.

        Args:
            df_target (pd.DataFrame): DataFrame that needs the target column updated.
            df_map (pd.DataFrame): DataFrame containing the protein-to-target mapping.

        Returns:
            pd.DataFrame: Updated df_target with mapped target values.
        """
        # Create a mapping dictionary {target -> protein}
        target_to_protein = dict(zip(df_map["target"], df_map["protein"]))

        # ƒApply mapping to create the "protein" column in df_target
        df_target["protein"] = df_target["target"].map(target_to_protein)

        return df_target

    binding_df_v2 = pd.read_csv(BINDING_DATA_PATH_v2)
    binding_df_v3 = pd.read_csv(BINDING_DATA_PATH_v3)

    binding_df = map_protein_to_target(binding_df_v3, binding_df_v2)

    return flex_data, binding_df, invalid_graphs

def load_data_v4():
    """Load glycan flexibility and binding data, process graphs, and return results."""
    flex_data = load_data_pdb()
    invalid_graphs = [glycan for glycan in flex_data if not isinstance(flex_data[glycan], nx.Graph)]
    binding_df = pd.read_csv(BINDING_DATA_PATH_v4)

    return flex_data, binding_df, invalid_graphs