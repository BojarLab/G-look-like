from scripts.find import find_matching_glycan
from scripts.process_with_motifs import process_glycan_with_motifs
from scripts.compute_attr import compute_stats
from glycowork.motif.processing import get_class
import networkx as nx
import numpy as np
from typing import Dict, List, Union, Any


def glycan_metric_dict__(properties: str,
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
        result: Dict[str, Union[str, List[float], bool]] = process_glycan_with_motifs(matched_glycan, properties, flex_data)

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
        result: Dict[str, Any] = process_glycan_with_motifs(matched_glycan, properties, flex_data)

        # Handle the new structure
        if 'glycan' in result:
            # New structure
            mono = result['glycan']['mono'].split(',')
            sasa = [result['glycan']['sasa']]  # Wrap in list for compute_stats
            flex = [result['glycan']['flex']]  # Wrap in list for compute_stats
            found_motifs = result['matched_motifs']

            # Extract per-motif data for all monosaccharides
            all_monos = []
            all_sasa = []
            all_flex = []
            for motif_data in result['per_motif']:
                motif_monos = motif_data['mono'].split(',')
                all_monos.extend(motif_monos)
                if 'sasa' in motif_data:
                    all_sasa.append(motif_data['sasa'])
                if 'flex' in motif_data:
                    all_flex.append(motif_data['flex'])

        else:
            # Old structure
            mono = result['combined']['monosaccharides']
            sasa = result['combined']['sasa_values']
            flex = result['combined']['flexibility_values']
            found_motifs = result['matched_motifs']
            all_monos = mono
            all_sasa = sasa
            all_flex = flex

        mono = [m for m in mono if m.strip()]

        if mono:
            metric_data.append({
                "glycan": glycan,
                "binding_score": binding_score,
                "mean_SASA": compute_stats(sasa, "SASA")["SASA_mean"],
                "sum_SASA": compute_stats(sasa, "SASA")["SASA_sum"],
                "max_SASA": compute_stats(sasa, "SASA")["SASA_max"],
                "mean_flex": compute_stats(flex, "flex")["flex_mean"],
                "sum_flex": compute_stats(flex, "flex")["flex_sum"],
                "max_flex": compute_stats(flex, "flex")["flex_max"],
                "match_mono": mono,
                "all_monos": all_monos,  # Include all monos from all motifs
                "motifs": found_motifs,
                "class": get_class(matched_glycan) or np.nan,

                # Optionally, you can store the per-motif data for more detailed analysis later
                "per_motif_data": result['per_motif']
            })

    print(f"Processed {len(metric_data)} glycans with metrics.")

    if missing_glycans:
        print(f"Not-matched glycan in flex data: {len(missing_glycans)}")

    return metric_data