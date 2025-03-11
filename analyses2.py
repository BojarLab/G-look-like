from functions2c import metric_df,compare_agg_methods, find_best_combo

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
        "motif": ["Fuc(a1-3)[Gal(b1-4)]GlcNAc",],
        "termini_list": [["t", "f", "f"]]
    },
    "LTA": {
        "motif": ["Fuc(a1-3)[Gal(b1-4)]GlcNAc",],
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
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ],
        "termini_list": [["t", "f", "f"] ]
    },
    "PTL-II": {
        "motif": ["Fuc(a1-2)Gal(b1-4)GlcNAc" ],
        "termini_list": [["t", "f", "f"] ]
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
        "motif": ["Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)", "Fuc(a1-2)Gal(b1-3)GalNAc(b1-4)[Sia(a2-3)]Gal(b1-4)GlcNAc(b1-3)"],
        "termini_list": [["t", "f", "t", "f", "f"], ["t", "f", "f", "t", "f", "f"]]

    },

    "MAL-I": {
        "motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },
    "MAA": {
        "motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
        "termini_list": [["t", "f"], ["t", "f"], ["t", "f", "f"]]
    },
    "MAL": {
        "motif": ["Gal3S(b1-4)GlcNAc", "Gal3S(b1-4)GlcNAc6S", "Sia(a2-3)Gal(b1-4)GlcNAc"],
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
        "motif": ["GlcNAc(b1-3)", "Fuc(a1-2)Gal(b1-4)GlcNAc", "Fuc(a1-2)Gal(b1-3)GalNAc" ],
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
    }
}
glycan_metrics = {}
test_glycans = {}
for lectin, properties in lectin_binding_motif.items():
    glycan_metric, test_glycan = metric_df(lectin, properties)
    if not glycan_metric.empty:
        glycan_metrics[lectin] = glycan_metric
    if test_glycan:
        test_glycans[lectin] = test_glycan

agg_df = compare_agg_methods(test_glycans)




