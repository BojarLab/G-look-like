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
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },

    "MAA": {
        "motif": ["Gal(b1-4)GlcNAc", "Gal(b1-4)GlcNAc6S", "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },
    "MAL": {
        "motif": ["Gal(b1-4)GlcNAc", "Gal(b1-4)GlcNAc6S", "Neu5Ac(a2-3)Gal(b1-4)GlcNAc"],
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
        "termini_list": [["t"], ["t", "f"]]},

    "AAA": {
    "motif": ["Fuc(a1-2)"],
    "termini_list": [["t"]]
    }
    ,
    "WFL": {
        "motif": ["GalNAc", "Gal(b1-3/4)GlcNAc(b1-3)"],
        "termini_list": [["t"], ["t", "f"]]
    },
    "ASA": {
        "motif": ["Man(a1-2)"],
        "termini_list": [["f"]]
    },

    ""
    "BambL": {
        "motif": ["Fuc(a1-?)"],
        "termini_list": [["t"]]
    },
    "Ricin B chain": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },
    "Ricin B": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },
    "RCA II": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },
    "RCA-II": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },
    "RCA60": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },
    "RCA-60": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    },
    "Ricin": {
        "motif": ["Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"]]
    }   ,

    "SSA": {
        "motif": ["GalNAc(a1-?)",],
        "termini_list": [["t"]]
    },
    "SSL": {
        "motif": ["GalNAc(a1-?)", ],
        "termini_list": [["t"]]
    },

    #more

"AIA": {
    "motif": ["Gal"],
    "termini_list": [["t"]]
  },
  "PL": {
    "motif": ["GlcNAc"],
    "termini_list": [["t"]]
  },
  "BPA": {
    "motif": ["Gal"],
    "termini_list": [["t"]]
  },
  "ECA": {
    "motif": ["Gal", "GalNAc"],
    "termini_list": [["t"], ["t"]]
  },
  "VVA": {
    "motif": ["GalNAc"],
    "termini_list": [["t"]]
  },
  "DC-SIGN": {
    "motif": ["Man"],
    "termini_list": [["t"]]
  },
  "LCA": {
    "motif": ["Glc", "Man"],
    "termini_list": [["t"], ["t"]]
  },
  "ACG": {
    "motif": ["Gal"],
    "termini_list": [["t"]]
  },
  "Artocarpin": {
    "motif": ["Glc", "Man"],
    "termini_list": [["t"], ["t"]]
  },
  "Heltuba": {
    "motif": ["Glc", "Man"],
    "termini_list": [["t"], ["t"]]
  },
  "CRLL": {
    "motif": ["Man"],
    "termini_list": [["t"]]
  },
  "GS-II": {
    "motif": ["GlcNAc"],
    "termini_list": [["t"]]
  },
  "CEL-IV": {
    "motif": ["Gal", "GalNAc"],
    "termini_list": [["t"], ["t"]]
  },
  "ConA": {
    "motif": ["Glc", "Man"],
    "termini_list": [["t"], ["t"]]
  },
  "DBA": {
    "motif": ["Gal", "GalNAc"],
    "termini_list": [["t"], ["t"]]
  },
  "PHA-E": {
    "motif": ["GalNAc"],
    "termini_list": [["t"]]
  },
  "CCA": {
    "motif": ["Glc", "Man"],
    "termini_list": [["t"], ["t"]]
  },
  "LSECtin": {
    "motif": ["Man"],
    "termini_list": [["t"]]
  },
  "BRA3": {
    "motif": ["Gal", "Man"],
    "termini_list": [["t"], ["t"]]
  },
  "BRA2": {
    "motif": ["Gal", "Man"],
    "termini_list": [["t"], ["t"]]
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
    "SSL": "SSA",
    "BambL": "BambL",
    "Ricin B chain": "RCA-II",
    "Ricin B": "RCA-II",
    "RCA II": "RCA-II",
    "RCA-II": "RCA-II",
    "RCA60": "RCA-II",
    "RCA-60": "RCA-II",
    "Ricin": "RCA-II",
    "ASA ": "ASA",
#more
  # Jacalin/Artocarpus group
  "Artocarpus Integrifolia Agglutinin": "AIA",
  "AIA": "AIA",

  # PL group
  "PL": "PL",
  "STL": "PL",
  "Solanum Tuberosum Lectin": "PL",

  # BPA group
  "BPA": "BPA",
  "BPL": "BPA",
  "Bauhinia Purpurea Lectin": "BPA",

  # ECA group
  "ECA": "ECA",
  "ECL": "ECA",
  "Erythrina Crista-galli Lectin": "ECA",

  # VVA group
  "VVA": "VVA",
  "VVL": "VVA",
  "VVA-G": "VVA",
  "Vicia Villosa Lectin": "VVA",

  # DC-SIGN group
  "DC-SIGN": "DC-SIGN",
  "LSECtin CRD Fc": "DC-SIGN",

  # LCA group
  "LCA": "LCA",
  "LcH": "LCA",
  "Lens Culinaris Agglutinin": "LCA",

  # ACG group
  "ACG": "ACG",
  "Agrocybe Cylindracea Galectin": "ACG",

  # Artocarpin group
  "Artocarpin": "Artocarpin",
  "Mannose-specific lectin KM+": "Artocarpin",

  # Heltuba group
  "Heltuba": "Heltuba",
  "Lectin 1": "Heltuba",

  # CRLL group
  "CRLL": "CRLL",

  # GS-II group
  "GS-II": "GS-II",
  "GSL-II": "GS-II",
  "Griffonia Simplicifolia Lectin 2": "GS-II",
  "Insecticidal N-acetylglucosamine-specific lectin (Fragment)": "GS-II",

  # CEL-IV group
  "CEL-IV": "CEL-IV",
  "Lectin CEL-IV, C-type": "CEL-IV",

  # ConA group
  "Canavalia Ensiformis Agglutinin": "ConA",
  "ConA": "ConA",
  "Concanavalin-A": "ConA",

  # DBA group
  "DBA": "DBA",
  "DBL": "DBA",
  "Dolichos Biflorus Agglutinin": "DBA",
  "Seed lectin subunit I": "DBA",

  # PHA-E group
  "PHA-E": "PHA-E",
  "Phaseolus Vulgaris Erythroagglutinin": "PHA-E",
  "Erythroagglutinating phytohemagglutinin": "PHA-E",

  # CCA group
  "CCA": "CCA",
  "Agglutinin": "CCA",

  # LSECtin group
  "LSECtin": "LSECtin",

  # BRA-2 group
  "BRA-2": "BRA2",
  "BRA2": "BRA2",
  "Lectin BRA-2": "BRA2",

  # BRA-3 group
  "BRA-3": "BRA3",
  "BRA3": "BRA3",
  "Lectin BRA-3": "BRA3"
}


structure_graphs, glycan_binding, invalid_graphs = load_data_v4()
oracle= pd.read_csv('data/all_meta.csv')

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

for l in lectin_binding_motif:
        if l in  binding_data_filt.columns:
            plot_correlation(l, binding_data_filt, f'results/plots/{l}.pdf')

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

get_lectin_clusters(binding_data_filt, filepath='results/plots/cluster.pdf')"""


def get_lectin_clusters(binding_data, oracle, filepath='', agg1=np.nanmean, agg2=np.nanmean):
    sasa_df, flex_df = pd.DataFrame(index=glycan_dict.keys()), pd.DataFrame(index=glycan_dict.keys())
    for lectin in binding_data.columns:
        binding_motif = lectin_binding_motif[lectin]
        motif_graphs = [glycan_to_nxGraph(binding_motif['motif'][i], termini='provided',
                                          termini_list=binding_motif['termini_list'][i]) for i in
                        range(len(binding_motif['motif']))]
        all_sasa, all_flex = [], []
        for _, ggraph in glycan_dict.items():
            _, matches = zip(
                *[subgraph_isomorphism(ggraph, motif_graph, return_matches=True) for motif_graph in motif_graphs])
            matches_unwrapped = unwrap(matches)
            all_sasa.append(agg2([agg1([ggraph.nodes()[n].get('SASA', np.nan) for n in m]) for m in
                                  matches_unwrapped]) if matches_unwrapped else np.nan)
            all_flex.append(agg2([agg1([ggraph.nodes()[n].get('flexibility', np.nan) for n in m]) for m in
                                  matches_unwrapped]) if matches_unwrapped else np.nan)
        sasa_df[lectin] = all_sasa
        flex_df[lectin] = all_flex
    sasa_corr = sasa_df.corrwith(binding_data)
    flex_corr = flex_df.corrwith(binding_data)
    out = pd.concat([sasa_corr, flex_corr], axis=1)
    out.columns = ['SASA_corr', 'flex_corr']

    # Add lectin_class from oracle
    lectin_classes = {}
    for lectin in out.index:
        # Check if the lectin index is in oracle's name column
        matches = oracle[oracle["name"] == lectin]
        if not matches.empty:
            lectin_classes[lectin] = matches.iloc[0]["lectin_class"]
        else:
            lectin_classes[lectin] = "Unknown"

    out['lectin_class'] = pd.Series(lectin_classes)

    fig, ax = plot_correlation_scatter(out.dropna(), filepath=filepath)

    return out, fig, ax  # Return the dataframe and plot objects


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

    # Create figure for main plot only
    fig, ax = plt.subplots(figsize=(12, 10))

    # Select color scheme - use lectin_class if available
    if 'lectin_class' in out.columns and out['lectin_class'].nunique() > 1:
        palette = plt.cm.tab10 if out['lectin_class'].nunique() <= 10 else plt.cm.tab20
        colors = {cls: palette(i % palette.N) for i, cls in enumerate(out['lectin_class'].unique())}
        hue_column = "lectin_class"
        palette_arg = colors
    else:
        hue_column = "quadrant"
        palette_arg = dict(zip(out["quadrant"].unique(), out["color"].unique()))

    # Create scatter plot
    sns.scatterplot(
        x="SASA_corr", y="flex_corr", data=out, s=120,
        hue=hue_column, palette=palette_arg, ax=ax,
        legend=False  # No legend in the main plot
    )

    # Add plot elements
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Correlation between SASA and Flexibility", fontsize=16)
    ax.set_xlabel("SASA Correlation", fontsize=14)
    ax.set_ylabel("Flexibility Correlation", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add point labels
    texts = [ax.annotate(idx, (row["SASA_corr"], row["flex_corr"]),
                         fontsize=10, fontweight="bold", color='black')
             for idx, row in out.iterrows()]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))

    # Add quadrant coloring
    ax.fill_between([-1, 0], 0, 1, alpha=0.1, color="royalblue")
    ax.fill_between([0, 1], 0, 1, alpha=0.1, color="darkgreen")
    ax.fill_between([-1, 0], -1, 0, alpha=0.1, color="darkred")
    ax.fill_between([0, 1], -1, 0, alpha=0.1, color="darkorange")
    ax.set_xlim(-0.8, 1.1)
    ax.set_ylim(-0.8, 1.1)

    # Add correlation coefficient
    corr_coef = np.corrcoef(out["SASA_corr"], out["flex_corr"])[0, 1]
    ax.annotate(f"Correlation: {corr_coef:.2f}", xy=(0.05, 0.95), xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8), fontsize=12)

    # Finalize layout
    plt.tight_layout()

    # Save main plot
    if filepath:
        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)

    # Create and save legend as a separate file
    if 'lectin_class' in out.columns and filepath:
        legend_title = "Lectin Class" if 'lectin_class' in out.columns else "Quadrants"
        legend_fig, legend_ax = plt.subplots(figsize=(3, 5))
        legend_ax.axis('off')

        # Create proxy artists for the legend
        from matplotlib.lines import Line2D
        legend_elements = []
        for cls, color in colors.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                          markersize=10, label=cls))

        # Add the legend to the new figure
        legend = legend_ax.legend(handles=legend_elements, title=legend_title,
                                  loc='center', fontsize=10, title_fontsize=12)

        # Save legend figure
        legend_filepath = filepath.replace('.pdf', '_legend.pdf')
        legend_fig.tight_layout()
        legend_fig.savefig(legend_filepath, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(legend_fig)

    return fig, ax
# Example usage
out, fig, ax = get_lectin_clusters(binding_data_filt, oracle, filepath='results/plots/cluster_.pdf')