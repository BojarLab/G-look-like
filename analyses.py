from scripts.process_lectins import process_lectin_motifs
from scripts.analyse_agg import analyze_binding_correlations, collect_correlation_statistics
import logging
import re
import pandas as pd
import numpy as np
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Outputs to console
        # Optional file handler:
        # logging.FileHandler('glycan_processing.log')
    ]
)



lectin_binding_motif = {
    "AOL": { "motif": ["Fuc(a1-?)"],
             "termini_list": [["t"]] },
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
        "motif": ["GlcNAc(b1-2/4)Man(a1-?)","Man(a1-2)"],
        "termini_list": [["t","f"], ["t"]]
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
        "termini_list": [["f", "f","f","f", "f"]]
    },
    "CA": {
        "motif": ["Gal(b1-3/4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-3/4)GlcNAc(b1-2)Man(a1-6)]Man"],
        "termini_list": [["f", "f","f","f", "f", "f", "f"]]
    },
    "CAA": {
        "motif": ["Gal(b1-3/4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-3/4)GlcNAc(b1-2)Man(a1-6)]Man"],
        "termini_list": [["f", "f","f","f", "f", "f", "f"]]
    },
    "TL": {
        "motif": ["GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"],
        "termini_list": [["f", "f","f","f", "f", "f", "f", "f"]]
    },
    "ACA": {
        "motif": ["Gal(b1-3)GalNAc", "GalOS(b1-3)GalNAc"], #OS??
        "termini_list": [["f", "f"], ["f", "f"]]
    },




    "AIA": {
        "motif": ["Gal(b1-3)GalNAc", "GlcNAC(b1-3)GalNAc"],
        "termini_list": [["f"], ["f"]]
    },
    "CF": {
        "motif": ["GalNAc", "GalNAc",  "GlcNAc", "GlcNAc"], #repeat is not needed
        "termini_list": [["t", "f", "t", "f"]]
    },
    "HAA": {
        "motif": ["GalNAc", "GlcNAc"], # linkage is missing
        "termini_list": [["t", "t"]]
    },
    "MPA": {
        "motif": ["Gal(b1-3)GlcNAc", "GlcNAc(b1-3)GalNAc"],
        "termini_list": [["f", "f"], ["t", "t"]]
    },
    "LAA": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc"], #why only fuc is terminal and the  ain brach is flexible?
        "termini_list": [["t", "t", "t"]] #t,f,f
    },
    "LcH": {
        "motif": ["Man(a1-6)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc", "Man(a1-2)Man"],
        "termini_list": [["f", "f", "f", "f", "t", "f"], ["t", "t"]]
    },
    "LTL": {
        "motif": ["Gal(b1-4)[Fuc(a1-3)]GlcNAc", "Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]GlcNAc]"], #can we put fuc or gal in branch?
        "termini_list": [["t", "t", "t"], ["t", "t", "t", "t"]]
    },
    "LTA": {
        "motif": ["Gal(b1-4)[Fuc(a1-3)]GlcNAc", "Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]GlcNAc]"],
        "termini_list": [["t", "t", "t"], ["t", "t", "t", "t"]]
    },
    "PSA": {
        "motif": ["Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"],
        "termini_list": [["f", "f", "f", "f", "t","f"]]
    },



    "PTL-I": {
        "motif": ["GalNAc(a1-3)Gal(a1-2)Fuc", "Gal(a1-3)Gal(a1-2)Fuc"], #why not just one main branch?
        "termini_list": [["t", "t", "t"], ["t", "t", "t"]]
    },
    "PTA-I": {
        "motif": ["GalNAc(a1-3)Gal(a1-2)Fuc", "Gal(a1-3)Gal(a1-2)Fuc"],
        "termini_list": [["t", "t", "t"], ["t", "t", "t"]]
    },


    "PTA-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ], #basically only Fuc is temrinal
        "termini_list": [["t", "t", "t"] ] #PTA has only one binding motif so why flexible?
    },
    "PTL-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ],
        "termini_list": [["t", "t", "t"] ] #same here why flexible?
    },


    "TJA-II": {
        "motif": ["Fuc(a1-2)Gal(b1-3)GalNAc", "Fuc(a1-2)Gal(b1-3/4)GlcNAc"],
        "termini_list": [["t", "t", "t"], ["t", "t", "t"]]
    },
    "UEA-I": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc", "Fuc(a1-2)Gal(b1-4)Glc","Fuc(a1-2)Gal(b1-4)[Fuc(a1-3)]GlcNAc" ],
        "termini_list": [["t", "t", "t"],["t", "t", "t"],["t", "t", "t"]]
    },
    "CTB": {
        "motif": ["Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)", "Fuc(a1-2)Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)"],
        "termini_list": [["t", "t", "t", "t", "t"], ["t", "t", "t", "t", "t", "t"]]

    },

    "MAL-I": {
        "motif": ["3SGal(b1-4)GlcNAc", "Sia(a2-3)Gal(b1-4)GlcNAc"], #3S, +-6S
        "termini_list": [["t", "t"], ["t", "t", "t"]]
    },
    "MAA": {
        "motif": ["3SGal(b1-4)GlcNAc", "Sia(a2-3)Gal(b1-4)GlcNAc"],  # 3S, +-6S
        "termini_list": [["t", "t"], ["t", "t", "t"]]
    },
    "MAL": {
        "motif": ["3SGal(b1-4)GlcNAc", "Sia(a2-3)Gal(b1-4)GlcNAc"],  # 3S, +-6S
        "termini_list": [["t", "t"], ["t", "t", "t"]]
    },


    "PSL": {
        "motif": ["Sia(a2-6)Gal(b1-3/4)GlcNAc"],
        "termini_list": [["t", "t", "t"]]
    },
    "TJA-I": {
        "motif": ["Sia(a2-6)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "t", "t"]]
    },



    "GS-II": {
        "motif": ["GlcNAc(b1-?)", "GlcNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },
    "PWA": {
        "motif": ["GlcNAc(b1-4)GlcNAc(b1-4)GlcNAc(b1-4)GlcNAc(b1-4)"],
        "termini_list": [["t", "t", "t", "t"]]
    },
    "UEA-II": {
        "motif": ["GlcNAc(b1-3)Gal", "GlcNAc(b1-3)GalNAc", "Fuc(a1-2)Gal(b1-4)GlCNAc", "Fuc(a1-2)Gal(b1-3)GalNAc(b1-3)" ],
        "termini_list": [["t", "t"], ["t", "t"], ["t", "t", "t"], ["t", "t", "t"]]
    },
    "WGA": {
        "motif": ["GlcNAc(b1-?)", "GlcNAc(a1-?)", "GalNAc(a/b1-?)", "Sia(a2-?)", "MurNAc(b1-?)"], #b1? or b2?
        "termini_list": [["t"], ["t"], ["t"], ["t"], ["t"]]
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
        "termini_list": [["t", "t"], ["t", "t"]]
    },
    "GS-I": {
        "motif": ["Gal(a1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },

    "LEA": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },
    "LEL": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },

    "MOA": {
        "motif": ["Gal(a1-3)Gal", "Gal(a1-3)GalNAc"], #Fuc is trimmed
        "termini_list": [["t", "t"], ["t", "f"]] #t,f or t,t,?
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
        "motif": ["Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)GlcNAc(b1-3)", "Gal(a1-3)[Fuc(a1-2)]Gal", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t", "t", "t"], ["t", "t", "t"], ["t", "t"]]
    },

    "STA": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },
    "STL": {
        "motif": ["Gal(b1-4)GlcNAc(b1-3)", "GlcNAc(b1-4)", "GalNAc(b1-4)GlcNAc"],
        "termini_list": [["t", "t"], ["t"], ["t", "t"]]
    },

    "CSA": {
        "motif": ["GalNAc(b1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t"], ["t"]]
    },
    "DBA": {
        "motif": ["GalNAc(a1-3)GalNAc(b1-?)"],
        "termini_list": [["t", "t"]]
    },
    "SBA": {
        "motif": ["GalNAc(b1-?)", "GalNAc(a1-3)GalNAc(b1-?)", "GalNAc(a1-?"],
        "termini_list": [["t"], ["t", "t"], ["t"]]
    },


    "VVL": {
        "motif": ["GalNAc(b1-3/4)GlcNAc", "GalNAc(b1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t", "f"], ["t"], ["t"]]
    },
    "VVA": {
        "motif": ["GalNAc(b1-3/4)GlcNAc", "GalNAc(b1-?)", "GalNAc(a1-?)"],
        "termini_list": [["t", "f"], ["t"], ["t"]]
    },


    "WFA": {
        "motif": ["GalNAc(b1-?)", "GalNAc(a1-?)", "GalNAc(b1-3/4)GlcNAc(b1-3)"],
        "termini_list": [["t"], ["t"], ["t", "t"]]
    }
}

# Process the metrics with specific aggregation methods
filtered_metrics, all_metrics = process_lectin_motifs(lectin_binding_motif, within=np.nansum, btw=np.nansum)

# Analyze binding correlations
correlation_results = analyze_binding_correlations(filtered_metrics)
correlation_results.to_excel('results/stats/lectin_binding_correlation_analysis_sum_sum.xlsx', index=False)

collect_correlation_statistics(filtered_metrics, within=np.nansum, btw=np.nansum)