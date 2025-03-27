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
        "termini_list": [["t", "t", "f"], ["t", "t", "f"]]

    },

    "PTA-I": {
        "motif": ["Fuc(a1-2)[GalNAc(a1-3)]Gal", "Fuc(a1-2)[Gal(a1-3)]Gal"],
        "termini_list": [["t", "t", "f"], ["t", "t", "f"]]

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

    "ECL": {
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

    },
    "WFL": {
        "motif": ["GalNAc", "Gal(b1-3/4)GlcNAc(b1-3)"],
        "termini_list": [["t"], ["t", "f"]]
    }
}
lectin_keys = {
    # Original entries
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
    "LTA": "LTA",
    "LTL": "LTA",
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
    "WFL": "WFL",

    # Additional entries from the list
    "ACL": "ACA",
    "Banana": "BanLec",
    "BanLec": "BanLec",
    "BambL": "BambL",
    "BC2L-A": "BC2L-A",
    "BC2LA": "BC2L-A",
    "BC2L-CN": "BC2L-CN",
    "BC2LCN": "BC2L-CN",
    "BDA": "BDA",
    "Calsepa": "Calsepa",
    "Cal sep A": "Calsepa",
    "Cholera toxin": "CTB",
    "Cholera toxin B": "CTB",
    "Con A": "ConA",
    "Concanavalin A": "ConA",
    "CVN": "CVN",
    "Cyanovirin-N": "CVN",
    "diCBM40": "diCBM40",
    "DSA": "DSA",
    "DSL": "DSA",
    "ECL": "ECL",
    "EEA": "EEA",
    "EEL": "EEA",
    "F17G": "F17G",
    "F17aG": "F17G",
    "F17bG": "F17G",
    "F17dG": "F17G",
    "F17eG": "F17G",
    "F17fG": "F17G",
    "GafD": "F17G",
    "GHA": "GHA",
    "GRFT": "GRFT",
    "griffithsin": "GRFT",
    "mGRFT": "GRFT",
    "GS I": "GS-I",
    "GSL I": "GS-I",
    "GSL-IB4": "GS-I",
    "GSL IB4": "GS-I",
    "GSL-I B4": "GS-I",
    "GS II": "GS-II",
    "GSL II": "GS-II",
    "AL": "HHL",
    "HPL": "HPA",
    "LBA": "LBA",
    "LBL": "LBA",
    "Lotus": "LTA",
    "MAA-I": "MAL-I",
    "MAA I": "MAL-I",
    "MAL I": "MAL-I",
    "MAA-II": "MAL-II",
    "MAA II": "MAL-II",
    "MAL II": "MAL-II",
    "MNA-G": "MNA-G",
    "MNA G": "MNA-G",
    "MNA M": "MNA-M",
    "MPL": "MPA",
    "OSA": "OSA",
    "OSL": "OSA",
    "Orysata": "OSA",
    "PA IL": "PA-IL",
    "Lec A": "PA-IL",
    "PA-IIL": "PA-IIL",
    "PA IIL": "PA-IIL",
    "LecB": "PA-IIL",
    "Lec B": "PA-IIL",
    "PHA": "PHA",
    "PHA-M": "PHA",
    "PHA M": "PHA",
    "PHA-P": "PHA",
    "PHA P": "PHA",
    "Blackbean": "PHA",
    "PHA-E": "PHA-E",
    "PHA E": "PHA-E",
    "E-PHA": "PHA-E",
    "E PHA": "PHA-E",
    "PHA-L": "PHA-L",
    "PHA L": "PHA-L",
    "L-PHA": "PHA-L",
    "L PHA": "PHA-L",
    "PhoSL": "PhoSL",
    "PEA": "PSA",
    "PSL1a": "PSL",
    "PTA I": "PTL-I",
    "PTA-GalNAc": "PTL-I",
    "PTA GalNAc": "PTL-I",
    "PTL I": "PTL-I",
    "PTL-GalNAc": "PTL-I",
    "PTL GalNAc": "PTL-I",
    "WBA-I": "PTL-I",
    "WBA I": "PTL-I",
    "PTA II": "PTL-II",
    "PTA-Gal": "PTL-II",
    "PTA Gal": "PTL-II",
    "PTL II": "PTL-II",
    "PTL-Gal": "PTL-II",
    "PTL Gal": "PTL-II",
    "WBA-II": "PTL-II",
    "WBA II": "PTL-II",
    "PWL": "PWA",
    "RCA 120": "RCA-I",
    "RCA I": "RCA-I",
    "Ricin B chain": "RCA-II",
    "Ricin B": "RCA-II",
    "RCA II": "RCA-II",
    "RCA-II": "RCA-II",
    "RCA60": "RCA-II",
    "RCA-60": "RCA-II",
    "Ricin": "RCA-II",
    "RPA": "RPA",
    "RPL": "RPA",
    "RSL": "RSL",
    "RS-IL": "RSL",
    "RS IL": "RSL",
    "RS-Fuc": "RSL",
    "RS-IIL": "RS-IIL",
    "RS IIL": "RS-IIL",
    "SBL": "SBA",
    "SLBR-B": "SLBR-B",
    "SLBR-H": "SLBR-H",
    "SLBR-N": "SLBR-N",
    "SNA-II": "SNA-II",
    "SNA II": "SNA-II",
    "SSA": "SSA",
    "SSL": "SSA",
    "PL": "STL",
    "SVN": "SVN",
    "Scytovirin": "SVN",
    "TJA I": "TJA-I",
    "UEA": "UEA-I",
    "UEA I": "UEA-I",
    "UEA II": "UEA-II",
    "WGL": "WGA"
}

structure_graphs, glycan_binding, invalid_graphs = load_data_v4()

binding_data = glycan_binding.set_index('protein').drop(['target'], axis=1)
idx = [k for k in binding_data.index if k in lectin_keys]
binding_data = binding_data.loc[idx,:]
binding_data.index = [lectin_keys[k] for k in binding_data.index]
binding_data = binding_data.groupby(level=0).median().dropna(axis=1, how='all').T
glycan_dict = {g:v for g,v in structure_graphs.items() if g in binding_data.index or any(compare_glycans(g, b) for b in binding_data.index)}


## filtered lectins for agg1 test
#lectins = {v for k,v in lectin_keys.items() if k in lectin_binding_motif and any(len(t)>1 for t in lectin_binding_motif[k]['termini_list'])
#           and k in glycan_binding.protein.tolist()}
lectins = {v for k,v in lectin_keys.items() if k in lectin_binding_motif and k in glycan_binding.protein.tolist()}
binding_data_filt = binding_data.loc[list(glycan_dict.keys()), list(lectins)]
print(binding_data_filt.shape)

def get_correlations(lectins, glycan_dict, binding_data, agg1=np.nanmean, agg2=np.nanmean):
  sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())
  for lectin in lectins:
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
  return sasa_corr, flex_corr


sasa_corr, flex_corr = zip(*[get_correlations(lectins, glycan_dict, binding_data_filt, agg2=k) for k in [np.nansum, np.nanmax, np.nanmean]])

sasa_corr = pd.concat(sasa_corr, axis=1)
sasa_corr.columns = ['sum', 'max', 'mean']
flex_corr = pd.concat(flex_corr, axis=1)
flex_corr.columns = ['sum', 'max', 'mean']


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

"""for l in lectin_specificity["abbreviation"]:
        if l in binding_data.columns:
            plot_correlation(l, binding_data_filt, f'results/plots/{l}.pdf')"""

for row in lectin_specificity["abbreviation"]:
    for l in row.split(","):
        l = l.strip()
        if l not in lectin_binding_motif.keys():
            print(l)
            continue