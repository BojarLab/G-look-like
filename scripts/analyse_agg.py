import numpy as np
import pandas as pd
from scipy import stats



def analyze_binding_correlations(filtered_metrics):
    """
    Analyze correlations between SASA/flexibility and binding scores across lectins.

    This function calculates:
    1. Pearson correlation coefficients
    2. Absolute t-statistics for correlation significance
    3. p-values for statistical significance
    4. Comparison of which property correlates more strongly with binding

    Parameters:
    -----------
    filtered_metrics : dict
        Dictionary of lectins with their corresponding metric DataFrames

    Returns:
    --------
    pd.DataFrame
        Summary of correlation statistics for each lectin
    """
    results = []

    for lectin, df in filtered_metrics.items():
        try:
            # Verify required columns exist
            if not all(col in df.columns for col in ['SASA', 'flexibility', 'binding_score']):
                print(f"Skipping {lectin}: Missing required columns")
                continue

            # Remove rows with any missing values in these columns
            valid_data = df[['SASA', 'flexibility', 'binding_score']].dropna()
            n_samples = len(valid_data)

            if n_samples < 3:  # Need at least 3 samples for meaningful correlation
                print(f"Skipping {lectin}: Insufficient samples (n={n_samples})")
                continue

            # Calculate Pearson correlations with p-values
            sasa_corr, sasa_pvalue = stats.pearsonr(
                valid_data['SASA'],
                valid_data['binding_score']
            )

            flex_corr, flex_pvalue = stats.pearsonr(
                valid_data['flexibility'],
                valid_data['binding_score']
            )

            # Calculate t-statistics for the correlations
            # Formula: t = r × sqrt((n-2)/(1-r²))
            sasa_tstat = sasa_corr * np.sqrt((n_samples - 2) / (1 - sasa_corr ** 2))
            flex_tstat = flex_corr * np.sqrt((n_samples - 2) / (1 - flex_corr ** 2))

            # Get absolute values
            sasa_tstat_abs = abs(sasa_tstat)
            flex_tstat_abs = abs(flex_tstat)
            sasa_corr_abs = abs(sasa_corr)
            flex_corr_abs = abs(flex_corr)

            # Store all results for this lectin
            results.append({
                'lectin': lectin,
                'sample_size': n_samples,
                # SASA statistics
                'sasa_binding_corr': sasa_corr,
                'sasa_binding_corr_abs': sasa_corr_abs,
                'sasa_pvalue': sasa_pvalue,
                'sasa_tstat_abs': sasa_tstat_abs,
                # Flexibility statistics
                'flex_binding_corr': flex_corr,
                'flex_binding_corr_abs': flex_corr_abs,
                'flex_pvalue': flex_pvalue,
                'flex_tstat_abs': flex_tstat_abs,
                # Comparison
                'stronger_correlation': 'SASA' if sasa_corr_abs > flex_corr_abs else 'Flexibility',
                'correlation_difference': sasa_corr_abs - flex_corr_abs,
                # Significance indicators
                'sasa_significant': sasa_pvalue < 0.05,
                'flex_significant': flex_pvalue < 0.05
            })
        except Exception as e:
            print(f"Error analyzing {lectin}: {e}")
            continue

    # Create DataFrame and sort by absolute t-statistic values
    if not results:
        print("No valid correlation results could be generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Sort by absolute t-statistic values (highest first)
    results_df = results_df.sort_values(by=['sasa_tstat_abs'], ascending=False)

    # Calculate summary statistics
    if len(results_df) > 0:
        print(f"\nAnalysis Summary:")
        print(f"Total lectins analyzed: {len(results_df)}")
        print(f"Lectins with significant SASA correlation: {results_df['sasa_significant'].sum()}")
        print(f"Lectins with significant flexibility correlation: {results_df['flex_significant'].sum()}")
        print(f"Lectins where SASA has stronger correlation: {(results_df['stronger_correlation'] == 'SASA').sum()}")
        print(
            f"Lectins where flexibility has stronger correlation: {(results_df['stronger_correlation'] == 'Flexibility').sum()}")

    return results_df


def collect_correlation_statistics(filtered_metrics, within, btw):
    """
    Collects SASA and flexibility correlation scores from lectin binding data
    and saves them to separate Excel files, including information about aggregation methods.

    Parameters:
    -----------
    filtered_metrics : dict
        Dictionary of lectins with their corresponding metric DataFrames
    within : function, optional
        Within-group aggregation function (default: np.nansum)
    btw : function, optional
        Between-group aggregation function (default: np.nanmean)

    Returns:
    --------
    tuple
        (sasa_correlations_df, flex_correlations_df) - DataFrames with correlation statistics
    """
    # Initialize lists to store results
    sasa_results = []
    flex_results = []

    # Get function names for documentation
    within_name = within.__name__ if hasattr(within, '__name__') else "custom"
    btw_name = btw.__name__ if hasattr(btw, '__name__') else "custom"

    # Analyze each lectin
    for lectin, df in filtered_metrics.items():
        try:
            # Check if required columns exist
            if not all(col in df.columns for col in ['SASA', 'flexibility', 'binding_score']):
                print(f"Skipping {lectin}: Missing required columns")
                continue

            # Remove rows with missing values in these columns
            valid_data = df[['SASA', 'flexibility', 'binding_score']].dropna()
            n_samples = len(valid_data)

            if n_samples < 3:  # Need at least 3 samples for meaningful correlation
                print(f"Skipping {lectin}: Insufficient samples (n={n_samples})")
                continue

            # Calculate Pearson correlations with p-values
            sasa_corr, sasa_pvalue = stats.pearsonr(
                valid_data['SASA'],
                valid_data['binding_score']
            )

            flex_corr, flex_pvalue = stats.pearsonr(
                valid_data['flexibility'],
                valid_data['binding_score']
            )

            # Get absolute correlation values
            sasa_corr_abs = abs(sasa_corr)
            flex_corr_abs = abs(flex_corr)

            # Store SASA correlation results
            sasa_results.append({
                'lectin': lectin,
                'sample_size': n_samples,
                'correlation_abs': sasa_corr_abs,
                'pvalue': sasa_pvalue,
                'significant': sasa_pvalue < 0.05,
                'within_agg': within_name,
                'between_agg': btw_name
            })

            # Store flexibility correlation results
            flex_results.append({
                'lectin': lectin,
                'sample_size': n_samples,
                'correlation_abs': flex_corr_abs,
                'pvalue': flex_pvalue,
                'significant': flex_pvalue < 0.05,
                'within_agg': within_name,
                'between_agg': btw_name
            })

        except Exception as e:
            print(f"Error analyzing {lectin}: {e}")
            continue

    # Create DataFrames and sort by absolute correlation values (highest first)
    sasa_df = pd.DataFrame(sasa_results)
    flex_df = pd.DataFrame(flex_results)

    if not sasa_df.empty:
        sasa_df = sasa_df.sort_values(by='correlation_abs', ascending=False)

    if not flex_df.empty:
        flex_df = flex_df.sort_values(by='correlation_abs', ascending=False)

    # Print summary statistics
    print("\n=== CORRELATION ANALYSIS SUMMARY ===")
    print(f"Aggregation methods: within={within_name}, between={btw_name}")
    print(f"Total lectins analyzed: {len(sasa_df)}")
    print(f"SASA correlations: {len(sasa_df)} lectins, {sasa_df['significant'].sum()} significant")
    print(f"Flexibility correlations: {len(flex_df)} lectins, {flex_df['significant'].sum()} significant")

    # Calculate average correlation values
    if not sasa_df.empty:
        avg_sasa_corr = sasa_df['correlation_abs'].mean()
        print(f"Average absolute SASA correlation: {avg_sasa_corr:.3f}")

    if not flex_df.empty:
        avg_flex_corr = flex_df['correlation_abs'].mean()
        print(f"Average absolute flexibility correlation: {avg_flex_corr:.3f}")

    # Compare which property correlates more strongly on average
    if not sasa_df.empty and not flex_df.empty:
        if avg_sasa_corr > avg_flex_corr:
            print(f"SASA correlates more strongly on average (by {avg_sasa_corr - avg_flex_corr:.3f})")
        else:
            print(f"Flexibility correlates more strongly on average (by {avg_flex_corr - avg_sasa_corr:.3f})")

    # Save to Excel files with aggregation method information
    sasa_excel_path = f'results/stats/sasa_correlations_{within_name}_{btw_name}.xlsx'
    flex_excel_path = f'results/stats/flexibility_correlations_{within_name}_{btw_name}.xlsx'

    if not sasa_df.empty:
        sasa_df.to_excel(sasa_excel_path, index=False)
        print(f"\nSASA correlations saved to: {sasa_excel_path}")

    if not flex_df.empty:
        flex_df.to_excel(flex_excel_path, index=False)
        print(f"Flexibility correlations saved to: {flex_excel_path}")

    return sasa_df, flex_df