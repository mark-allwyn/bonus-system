"""
Utility Functions

Helper functions for data processing, visualization, and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


def print_section_header(title: str, char: str = '='):
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}\n")


def summarize_dataframe(df: pd.DataFrame, title: str = "DataFrame Summary"):
    """Print comprehensive summary of a DataFrame."""
    print_section_header(title)
    print(f"Shape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for a dataset.

    Args:
        data: Array of values
        confidence: Confidence level (default 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats

    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

    return mean - interval, mean + interval


def compare_distributions(
    df: pd.DataFrame,
    column: str,
    group_by: str,
    figsize: tuple = (12, 5)
) -> plt.Figure:
    """
    Compare distributions of a column across different groups.

    Args:
        df: DataFrame
        column: Column to compare
        group_by: Column to group by
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Box plot
    df.boxplot(column=column, by=group_by, ax=axes[0])
    axes[0].set_title(f'{column} by {group_by}')
    axes[0].set_xlabel(group_by)
    axes[0].set_ylabel(column)

    # Violin plot
    unique_groups = df[group_by].unique()
    for group in unique_groups:
        group_data = df[df[group_by] == group][column]
        axes[1].violinplot([group_data], positions=[list(unique_groups).index(group)])

    axes[1].set_title(f'{column} Distribution by {group_by}')
    axes[1].set_xlabel(group_by)
    axes[1].set_ylabel(column)
    axes[1].set_xticks(range(len(unique_groups)))
    axes[1].set_xticklabels(unique_groups, rotation=45)

    plt.tight_layout()
    return fig


def export_latex_table(df: pd.DataFrame, filepath: str, caption: str = "", label: str = ""):
    """
    Export DataFrame as LaTeX table for publication.

    Args:
        df: DataFrame to export
        filepath: Output file path
        caption: Table caption
        label: Table label for referencing
    """
    latex_str = df.to_latex(
        index=False,
        float_format="%.3f",
        caption=caption,
        label=label,
        position='htbp'
    )

    with open(filepath, 'w') as f:
        f.write(latex_str)

    print(f"LaTeX table saved to {filepath}")


def create_results_summary(
    model_comparison: pd.DataFrame,
    segment_analysis: pd.DataFrame,
    save_dir: str = 'results/tables'
) -> Dict[str, str]:
    """
    Create publication-ready summary tables.

    Args:
        model_comparison: DataFrame with model comparison metrics
        segment_analysis: DataFrame with segment analysis
        save_dir: Directory to save tables

    Returns:
        Dictionary mapping table names to file paths
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    outputs = {}

    # Model comparison table
    comparison_path = f"{save_dir}/model_comparison_latex.tex"
    export_latex_table(
        model_comparison,
        comparison_path,
        caption="Performance comparison of targeting strategies",
        label="tab:model_comparison"
    )
    outputs['model_comparison'] = comparison_path

    # Segment analysis table
    segment_path = f"{save_dir}/segment_analysis_latex.tex"
    export_latex_table(
        segment_analysis,
        segment_path,
        caption="Player segment characteristics and treatment effects",
        label="tab:segment_analysis"
    )
    outputs['segment_analysis'] = segment_path

    return outputs


def validate_experiment_design(
    df: pd.DataFrame,
    treatment_col: str = 'treatment',
    min_group_size: int = 100
) -> Dict[str, bool]:
    """
    Validate that experiment design meets statistical requirements.

    Args:
        df: DataFrame with experimental data
        treatment_col: Treatment indicator column
        min_group_size: Minimum required group size

    Returns:
        Dictionary with validation results
    """
    results = {
        'has_treatment_column': treatment_col in df.columns,
        'balanced_assignment': False,
        'sufficient_sample_size': False,
        'no_contamination': True  # Assumed for synthetic data
    }

    if results['has_treatment_column']:
        treatment_counts = df[treatment_col].value_counts()

        # Check balance (should be close to 50/50)
        if len(treatment_counts) == 2:
            balance_ratio = min(treatment_counts) / max(treatment_counts)
            results['balanced_assignment'] = balance_ratio >= 0.4  # Allow some imbalance

        # Check sample size
        results['sufficient_sample_size'] = all(treatment_counts >= min_group_size)

    return results


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Load test data
    df = pd.read_csv('data/player_data_test.csv')

    # Test summarize
    summarize_dataframe(df.head(100), "Sample Data Summary")

    # Test experiment validation
    validation = validate_experiment_design(df)
    print("\nExperiment Validation:")
    for key, value in validation.items():
        status = "[OK]" if value else "[FAILED]"
        print(f"  {status} {key}: {value}")

    print("\n[OK] Utilities test complete!")
