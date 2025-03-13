from scripts.metric import metric_df
import re
import numpy as np



def process_lectin_motifs(lectin_binding_motif, within, btw):
    """
    Process all lectins and filter those with multiple monosaccharides in their motifs.
    Returns two dictionaries: filtered and complete metrics.
    """
    # Initialize both dictionaries
    filtered_metrics = {}
    all_metrics = {}

    # Define pattern to match monosaccharides
    pattern = r'(Gal|Glc|Man|Fuc|Sia|GalNAc|GlcNAc|NeuAc)'

    # Process all lectins
    for lectin, properties in lectin_binding_motif.items():
        print(f"\nProcessing lectin: {lectin}")
        print(f"Motif: {properties['motif']}")

        # Get metrics for this lectin
        df = metric_df(lectin, properties, within, btw)

        if df.empty:
            print(f"⚠️ Skipping {lectin}: NO binding data.")
            continue

        # Add to the complete metrics dictionary
        all_metrics[lectin] = df

        # Check if any motif has multiple monosaccharides
        for motif in properties["motif"]:
            if isinstance(motif, str) and len(re.findall(pattern, motif)) > 1:
                filtered_metrics[lectin] = df
                print(f"Added {lectin} to filtered set (multiple monosaccharides detected)")
                break

    print(f"\nTotal lectins: {len(all_metrics)}")
    print(f"Filtered lectins with multiple monosaccharides: {len(filtered_metrics)}")

    return filtered_metrics, all_metrics