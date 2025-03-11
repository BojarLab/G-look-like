from scripts.match_glycan import find_matching_glycan
from scripts.process_with_motifs import process_glycan_with_motifs
from scripts.compute_attr import compute_stats
from glycowork.motif.processing import get_class
import networkx as nx
import numpy as np


def glycan_metric_dict(properties: str,
                       glycan_scores: dict,
                       flex_data: dict[str, nx.Graph]) -> list[dict]:
    """ Generate a metrics dictionary for a given lectin and its properties. """
    metric_data = []
    missing_glycans = []

    for glycan in glycan_scores:
        binding_score = glycan_scores[glycan]
        matched_glycan = find_matching_glycan(flex_data, glycan)

        if not matched_glycan:
            missing_glycans.append(glycan)
            continue

        # Process the matched glycan using the new function
        result = process_glycan_with_motifs(matched_glycan, properties, flex_data)

        # Extract values from the new dictionary structure
        match_mono = result['combined']['monosaccharides']
        SASA = result['combined']['sasa_values']
        flexibility = result['combined']['flexibility_values']
        found_motifs = result['matched_motifs']

        match_mono = [m for m in match_mono if m.strip()]

        if match_mono:
            metric_data.append({
                "glycan": glycan,
                "binding_score": binding_score,
                "mean_SASA": compute_stats(SASA, "SASA")["SASA_mean"],  # Note: Changed "sasa_mean" to "SASA_mean"
                "sum_SASA": compute_stats(SASA, "SASA")["SASA_sum"],  # Changed prefix to match
                "max_SASA": compute_stats(SASA, "SASA")["SASA_max"],
                "mean_flex": compute_stats(flexibility, "flex")["flex_mean"],
                "sum_flex": compute_stats(flexibility, "flex")["flex_sum"],
                "max_flex": compute_stats(flexibility, "flex")["flex_max"],
                "match_mono": match_mono,
                "motifs": found_motifs,
                "class": get_class(matched_glycan) or np.nan,

                # Optionally, you can store the per-motif data for more detailed analysis later
                "per_motif_data": result['per_motif']
            })

    print(f"Processed {len(metric_data)} glycans with metrics.")

    if missing_glycans:
        print(f"Not-matched glycan in flex data:")

    return metric_data