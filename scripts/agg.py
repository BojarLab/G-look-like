def aggregate_values(values, how='sum'):
    """
    Aggregate a list of values based on the specified method.

    Args:
        values (list): List of values to aggregate
        how (str): Aggregation method ('sum', 'mean', 'max')

    Returns:
        float: Aggregated value
    """
    if not values:
        return 0.0

    if how == 'sum':
        return sum(values)
    elif how == 'mean':
        return sum(values) / len(values)
    elif how == 'max':
        return max(values)
    else:
        raise ValueError(f"Unsupported aggregation method: {how}")


def process_node_attributes(node_attrs, agg_within='sum', agg_across='max'):
    """
    Process node attributes with two levels of aggregation:
    1. Within-motif: Aggregates multiple nodes within a single motif
    2. Across-motif: Aggregates results from multiple motifs

    Args:
        node_attrs (list): List of dictionaries with attributes per motif
        agg_within (str): Aggregation method within motifs ('sum', 'mean', 'max')
        agg_across (str): Aggregation method across motifs ('sum', 'mean', 'max')

    Returns:
        dict: Processed attributes with within-motif and across-motif aggregations
    """
    # Process within motifs first
    motif_aggregations = []

    # For each motif, aggregate its values if needed
    for attrs in node_attrs:
        # Check if we need within-motif aggregation (multiple values in this motif)
        has_multiple_values = isinstance(attrs.get('sasa', 0), list) or isinstance(attrs.get('flex', 0), list)

        motif_agg = {
            'mono': ','.join(attrs.get('monos', [])),
            'sasa': aggregate_values(attrs.get('sasa', 0), how=agg_within) if has_multiple_values else attrs.get('sasa', 0),
            'flex': aggregate_values(attrs.get('flex', 0), how=agg_within) if has_multiple_values else attrs.get('flex', 0)
        }
        motif_aggregations.append(motif_agg)

    # Check if we need across-motif aggregation (multiple motifs)
    needs_across_agg = len(motif_aggregations) > 1

    # Filter out zero values for aggregation
    sasa_motif_values = [agg['sasa'] for agg in motif_aggregations if agg.get('sasa', 0) > 0]
    flex_motif_values = [agg['flex'] for agg in motif_aggregations if agg.get('flex', 0) > 0]

    # Get all unique monosaccharides
    all_monos = [item for attrs in node_attrs for item in attrs.get('monos', [])]

    if needs_across_agg:
        # Perform across-motif aggregation
        glycan_agg = {
            'mono': ','.join(set(all_monos)),
            'sasa': aggregate_values(sasa_motif_values, how=agg_across) if sasa_motif_values else 0.0,
            'flex': aggregate_values(flex_motif_values, how=agg_across) if flex_motif_values else 0.0
        }
    else:
        # Single motif case - use the motif values directly
        glycan_agg = motif_aggregations[0] if motif_aggregations else {
            'mono': ','.join(set(all_monos)),
            'sasa': 0.0,
            'flex': 0.0
        }

    return {
        'per_motif': motif_aggregations,
        'glycan': glycan_agg
    }