"""
Evaluation Framework for Uplift Modeling

Implements uplift-specific metrics and visualizations:
- Uplift calibration plots
- Cumulative gain curves (Qini curves)
- AUUC (Area Under Uplift Curve)
- Comparative analysis vs. baselines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
from sklearn.metrics import roc_auc_score
import os


class UpliftEvaluator:
    """Evaluation metrics and visualizations for uplift models."""

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', dpi: int = 300):
        """
        Initialize evaluator.

        Args:
            style: Matplotlib style
            dpi: Figure DPI for high-quality outputs
        """
        self.style = style
        self.dpi = dpi
        plt.style.use(style)
        sns.set_palette("Set2")

    def calculate_actual_uplift_by_group(
        self,
        df: pd.DataFrame,
        n_bins: int = 10,
        predicted_uplift_col: str = 'uplift_score',
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome'
    ) -> pd.DataFrame:
        """
        Calculate actual uplift in each predicted uplift decile.
        This is used for calibration analysis.

        Args:
            df: DataFrame with predictions and outcomes
            n_bins: Number of bins for grouping
            predicted_uplift_col: Column with predicted uplift
            treatment_col: Treatment indicator column
            outcome_col: Outcome column

        Returns:
            DataFrame with predicted vs actual uplift by bin
        """
        df = df.copy()

        # Create bins based on predicted uplift
        df['uplift_bin'] = pd.qcut(df[predicted_uplift_col], q=n_bins, labels=False, duplicates='drop')

        results = []
        for bin_idx in sorted(df['uplift_bin'].unique()):
            bin_data = df[df['uplift_bin'] == bin_idx]

            # Calculate actual uplift in this bin
            treatment_group = bin_data[bin_data[treatment_col] == 1]
            control_group = bin_data[bin_data[treatment_col] == 0]

            if len(treatment_group) > 0 and len(control_group) > 0:
                actual_uplift = (
                    treatment_group[outcome_col].mean() -
                    control_group[outcome_col].mean()
                )
                predicted_uplift = bin_data[predicted_uplift_col].mean()

                results.append({
                    'bin': bin_idx,
                    'predicted_uplift': predicted_uplift,
                    'actual_uplift': actual_uplift,
                    'n_samples': len(bin_data),
                    'treatment_response': treatment_group[outcome_col].mean(),
                    'control_response': control_group[outcome_col].mean()
                })

        return pd.DataFrame(results)

    def plot_uplift_calibration(
        self,
        df: pd.DataFrame,
        n_bins: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create uplift calibration plot showing predicted vs actual uplift.

        Args:
            df: DataFrame with predictions and outcomes
            n_bins: Number of bins
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Calculate uplift by bin
        calibration_df = self.calculate_actual_uplift_by_group(df, n_bins=n_bins)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        # Plot predicted vs actual
        ax.scatter(
            calibration_df['predicted_uplift'],
            calibration_df['actual_uplift'],
            s=calibration_df['n_samples'] / 10,
            alpha=0.6,
            label='Actual uplift'
        )

        # Perfect calibration line
        min_val = min(calibration_df['predicted_uplift'].min(), calibration_df['actual_uplift'].min())
        max_val = max(calibration_df['predicted_uplift'].max(), calibration_df['actual_uplift'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect calibration', linewidth=2)

        ax.set_xlabel('Predicted Uplift', fontsize=12)
        ax.set_ylabel('Actual Uplift', fontsize=12)
        ax.set_title('Uplift Model Calibration', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Calibration plot saved to {save_path}")

        return fig

    def calculate_cumulative_gains(
        self,
        df: pd.DataFrame,
        score_col: str,
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        n_points: int = 100
    ) -> pd.DataFrame:
        """
        Calculate cumulative gains curve (Qini curve).

        Args:
            df: DataFrame with scores and outcomes
            score_col: Column to rank by
            treatment_col: Treatment indicator
            outcome_col: Outcome column
            n_points: Number of points for the curve

        Returns:
            DataFrame with cumulative gains at each percentile
        """
        df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)

        # Calculate step size
        step = max(1, len(df_sorted) // n_points)
        percentiles = []
        cumulative_uplifts = []

        for i in range(0, len(df_sorted), step):
            subset = df_sorted.iloc[:i+1]

            if len(subset) > 0:
                # Calculate cumulative uplift
                treatment_group = subset[subset[treatment_col] == 1]
                control_group = subset[subset[treatment_col] == 0]

                if len(treatment_group) > 0 and len(control_group) > 0:
                    treatment_effect = treatment_group[outcome_col].sum()
                    control_effect = control_group[outcome_col].sum()

                    # Scale control to match treatment group size
                    scale_factor = len(treatment_group) / len(control_group)
                    cumulative_uplift = treatment_effect - (control_effect * scale_factor)

                    percentiles.append((i + 1) / len(df_sorted) * 100)
                    cumulative_uplifts.append(cumulative_uplift)

        return pd.DataFrame({
            'percentile': percentiles,
            'cumulative_uplift': cumulative_uplifts
        })

    def plot_cumulative_gains(
        self,
        df: pd.DataFrame,
        models: Dict[str, str],
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cumulative gain curves for multiple models.

        Args:
            df: DataFrame with model predictions
            models: Dict mapping model name to score column
            treatment_col: Treatment indicator column
            outcome_col: Outcome column
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 7), dpi=self.dpi)

        # Add random baseline
        n_treatment = (df[treatment_col] == 1).sum()
        n_control = (df[treatment_col] == 0).sum()
        random_uplift = (
            df[df[treatment_col] == 1][outcome_col].sum() -
            df[df[treatment_col] == 0][outcome_col].sum() * (n_treatment / n_control)
        )

        # Plot each model
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        for (model_name, score_col), color in zip(models.items(), colors):
            gains = self.calculate_cumulative_gains(df, score_col, treatment_col, outcome_col)
            ax.plot(
                gains['percentile'],
                gains['cumulative_uplift'],
                label=model_name,
                linewidth=2.5,
                color=color
            )

        # Random baseline
        ax.plot([0, 100], [0, random_uplift], 'k--', label='Random', linewidth=2, alpha=0.6)

        ax.set_xlabel('% of Population Targeted', fontsize=12)
        ax.set_ylabel('Cumulative Incremental Conversions', fontsize=12)
        ax.set_title('Cumulative Gain Curves (Qini Curves)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Cumulative gains plot saved to {save_path}")

        return fig

    def calculate_auuc(
        self,
        df: pd.DataFrame,
        score_col: str,
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome'
    ) -> float:
        """
        Calculate Area Under Uplift Curve (AUUC).

        Args:
            df: DataFrame with scores and outcomes
            score_col: Column to rank by
            treatment_col: Treatment indicator
            outcome_col: Outcome column

        Returns:
            AUUC value
        """
        gains = self.calculate_cumulative_gains(df, score_col, treatment_col, outcome_col)

        # Normalize to [0, 1]
        x = gains['percentile'].values / 100
        y = gains['cumulative_uplift'].values

        # Calculate area using trapezoidal rule
        auuc = np.trapz(y, x)

        return auuc

    def compare_models(
        self,
        df: pd.DataFrame,
        models: Dict[str, str],
        treatment_col: str = 'treatment',
        outcome_col: str = 'outcome',
        budget_constraint: float = 0.3
    ) -> pd.DataFrame:
        """
        Compare multiple models on key metrics.

        Args:
            df: DataFrame with model predictions
            models: Dict mapping model name to score column
            treatment_col: Treatment indicator
            outcome_col: Outcome column
            budget_constraint: Proportion of population to target

        Returns:
            DataFrame with comparative metrics
        """
        results = []

        # Calculate random baseline
        n_target = int(len(df) * budget_constraint)
        random_sample = df.sample(n=n_target, random_state=42)
        random_treatment = random_sample[random_sample[treatment_col] == 1]
        random_control = random_sample[random_sample[treatment_col] == 0]

        if len(random_control) > 0:
            random_uplift = (
                random_treatment[outcome_col].mean() -
                random_control[outcome_col].mean()
            )
        else:
            random_uplift = 0

        for model_name, score_col in models.items():
            # Select top players by score
            top_players = df.nlargest(n_target, score_col)

            # Calculate metrics
            top_treatment = top_players[top_players[treatment_col] == 1]
            top_control = top_players[top_players[treatment_col] == 0]

            if len(top_control) > 0 and len(top_treatment) > 0:
                avg_uplift = (
                    top_treatment[outcome_col].mean() -
                    top_control[outcome_col].mean()
                )
            else:
                avg_uplift = 0

            # Calculate AUUC
            auuc = self.calculate_auuc(df, score_col, treatment_col, outcome_col)

            # Calculate improvement over random
            improvement = ((avg_uplift / random_uplift) - 1) * 100 if random_uplift != 0 else 0

            results.append({
                'model': model_name,
                'avg_uplift_top_30pct': avg_uplift,
                'improvement_vs_random': improvement,
                'auuc': auuc,
                'n_targeted': n_target
            })

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('avg_uplift_top_30pct', ascending=False)

        return comparison_df

    def plot_segment_distribution(
        self,
        df: pd.DataFrame,
        segment_col: str = 'segment',
        true_uplift_col: str = 'true_uplift',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the four player segments and their characteristics.

        Args:
            df: DataFrame with segment labels
            segment_col: Column with segment labels
            true_uplift_col: Column with true uplift values
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # Plot 1: Segment sizes
        segment_counts = df[segment_col].value_counts()
        axes[0, 0].bar(segment_counts.index, segment_counts.values, color=sns.color_palette("Set2"))
        axes[0, 0].set_title('Player Segment Distribution', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Players')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: True uplift by segment
        segment_uplift = df.groupby(segment_col)[true_uplift_col].mean()
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in segment_uplift.values]
        axes[0, 1].bar(segment_uplift.index, segment_uplift.values, color=colors)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0, 1].set_title('True Treatment Effect by Segment', fontweight='bold')
        axes[0, 1].set_ylabel('True Uplift')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Outcome rates by segment and treatment
        outcome_data = []
        for segment in df[segment_col].unique():
            segment_df = df[df[segment_col] == segment]
            treatment_rate = segment_df[segment_df['treatment'] == 1]['outcome'].mean()
            control_rate = segment_df[segment_df['treatment'] == 0]['outcome'].mean()
            outcome_data.append({
                'segment': segment,
                'treatment': treatment_rate,
                'control': control_rate
            })

        outcome_df = pd.DataFrame(outcome_data)
        x = np.arange(len(outcome_df))
        width = 0.35
        axes[1, 0].bar(x - width/2, outcome_df['treatment'], width, label='Treatment', color='steelblue')
        axes[1, 0].bar(x + width/2, outcome_df['control'], width, label='Control', color='coral')
        axes[1, 0].set_title('Outcome Rates by Segment', fontweight='bold')
        axes[1, 0].set_ylabel('Engagement Rate')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(outcome_df['segment'], rotation=45)
        axes[1, 0].legend()

        # Plot 4: Segment targeting recommendation
        recommendations = {
            'persuadables': 'TARGET\n(Positive ROI)',
            'sure_things': 'SKIP\n(Wasted Spend)',
            'lost_causes': 'SKIP\n(No Effect)',
            'sleeping_dogs': 'AVOID\n(Negative Effect)'
        }

        rec_data = []
        for segment in df[segment_col].unique():
            rec_data.append({
                'segment': segment,
                'recommendation': recommendations.get(segment, 'Unknown'),
                'count': (df[segment_col] == segment).sum()
            })

        rec_df = pd.DataFrame(rec_data)
        colors_rec = ['green', 'orange', 'gray', 'red']
        axes[1, 1].barh(rec_df['segment'], rec_df['count'], color=colors_rec)
        axes[1, 1].set_title('Targeting Recommendations', fontweight='bold')
        axes[1, 1].set_xlabel('Number of Players')

        # Add recommendation text
        for i, row in rec_df.iterrows():
            axes[1, 1].text(
                row['count'] / 2, i, row['recommendation'],
                ha='center', va='center', fontweight='bold', color='white'
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Segment distribution plot saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Test evaluation framework
    print("Testing evaluation framework...")

    # Load data with predictions
    test_df = pd.read_csv('data/player_data_test.csv')

    # Load models and generate predictions
    from features import FeatureEngineer
    from uplift_model import TLearner, ResponseModel

    fe = FeatureEngineer(scale_features=True)
    train_df = pd.read_csv('data/player_data_train.csv')
    fe.prepare_features(train_df, fit_scaler=True)

    uplift_model = TLearner.load('data/uplift_model.pkl')
    response_model = ResponseModel.load('data/response_model.pkl')

    X_test = fe.prepare_features(test_df, fit_scaler=False)
    test_df['uplift_score'] = uplift_model.predict_uplift(X_test)
    test_df['response_score'] = response_model.predict_response(X_test)

    # Initialize evaluator
    evaluator = UpliftEvaluator()

    # Create output directory
    os.makedirs('results/figures', exist_ok=True)

    # Generate plots
    print("\nGenerating calibration plot...")
    evaluator.plot_uplift_calibration(
        test_df,
        save_path='results/figures/uplift_calibration.png'
    )

    print("Generating cumulative gains plot...")
    evaluator.plot_cumulative_gains(
        test_df,
        models={
            'Uplift Model': 'uplift_score',
            'Response Model': 'response_score',
            'True Uplift': 'true_uplift'
        },
        save_path='results/figures/cumulative_gains.png'
    )

    print("Generating segment distribution plot...")
    evaluator.plot_segment_distribution(
        test_df,
        save_path='results/figures/segment_distribution.png'
    )

    # Generate comparison metrics
    print("\nComparing models...")
    comparison_df = evaluator.compare_models(
        test_df,
        models={
            'Uplift Model': 'uplift_score',
            'Response Model': 'response_score',
            'True Uplift (Oracle)': 'true_uplift'
        }
    )

    print("\n=== Model Comparison ===")
    print(comparison_df.to_string(index=False))

    # Save comparison table
    os.makedirs('results/tables', exist_ok=True)
    comparison_df.to_csv('results/tables/model_comparison.csv', index=False)

    print("\n[OK] Evaluation test complete!")
    plt.close('all')
