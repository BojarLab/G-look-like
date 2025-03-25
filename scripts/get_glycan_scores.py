from typing import Dict

def get_glycan_scores(filtered_df: dict[str, float]) -> Dict[str, float]:
    """Calculate mean binding scores for glycans."""
    lectin_df = filtered_df.drop(columns=["target", "protein"])  # Exclude "protein" and "target" columns
    glycan_scores = lectin_df.mean(axis=0).to_dict()

    return glycan_scores
