"""
Scoring Pipeline for Player Prioritization

Combines uplift, LTV, and churn predictions with business rules
to generate composite priority scores for bonus allocation.

Priority Score = Uplift × LTV × Churn_Probability × Business_Rules
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import yaml


class ScoringPipeline:
    """
    Combines multiple model outputs to generate final priority scores.
    """

    def __init__(self, config_path: str = 'configs/simulation_config.yaml'):
        """Initialize scoring pipeline with business rules."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.business_rules = self.config['business_rules']

    def calculate_composite_score(
        self,
        uplift: np.ndarray,
        ltv: np.ndarray,
        churn_prob: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Calculate composite priority score.

        Args:
            uplift: Uplift predictions (treatment effect)
            ltv: Lifetime value predictions
            churn_prob: Churn probability predictions
            normalize: Whether to normalize scores to [0, 1]

        Returns:
            Composite scores (n_samples,)
        """
        # Ensure positive uplift (negative uplift = don't target)
        uplift_positive = np.maximum(uplift, 0)

        # Normalize LTV to [0, 1] for consistent scaling
        ltv_norm = (ltv - ltv.min()) / (ltv.max() - ltv.min() + 1e-8)

        # Composite score: uplift × LTV × churn risk
        # Higher score = higher priority for bonus
        scores = uplift_positive * ltv_norm * churn_prob

        if normalize:
            # Normalize final scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        return scores

    def apply_business_rules(
        self,
        df: pd.DataFrame,
        uplift_col: str = 'uplift_score',
        ltv_col: str = 'ltv_score',
        churn_col: str = 'churn_probability',
        composite_col: str = 'composite_score'
    ) -> pd.DataFrame:
        """
        Apply business rules to filter and flag players.

        Args:
            df: DataFrame with model predictions
            *_col: Column names for different scores

        Returns:
            DataFrame with business rule flags applied
        """
        df = df.copy()

        # Initialize eligibility flag
        df['eligible'] = True

        # Rule 1: Minimum uplift threshold
        min_uplift = self.business_rules.get('min_uplift_threshold', 0.1)
        df.loc[df[uplift_col] < min_uplift, 'eligible'] = False

        # Rule 2: Maximum churn risk (exclude very high churn players if configured)
        max_churn = self.business_rules.get('max_churn_risk', 0.9)
        df.loc[df[churn_col] > max_churn, 'eligible'] = False

        # Rule 3: Negative uplift = definitely exclude (Sleeping Dogs)
        df.loc[df[uplift_col] < 0, 'eligible'] = False

        return df

    def rank_players(
        self,
        df: pd.DataFrame,
        score_col: str = 'composite_score',
        eligible_col: str = 'eligible'
    ) -> pd.DataFrame:
        """
        Rank players by priority score.

        Args:
            df: DataFrame with scores
            score_col: Column to rank by
            eligible_col: Column indicating eligibility

        Returns:
            DataFrame with ranking
        """
        df = df.copy()

        # Rank all players (eligible only for actual targeting)
        df['rank_overall'] = df[score_col].rank(ascending=False, method='first')

        # Rank among eligible players
        df['rank_eligible'] = np.nan
        eligible_mask = df[eligible_col]
        if eligible_mask.sum() > 0:
            df.loc[eligible_mask, 'rank_eligible'] = df.loc[eligible_mask, score_col].rank(
                ascending=False, method='first'
            )

        return df

    def select_top_players(
        self,
        df: pd.DataFrame,
        budget_constraint: Optional[float] = None,
        score_col: str = 'composite_score',
        eligible_col: str = 'eligible'
    ) -> pd.DataFrame:
        """
        Select top players based on budget constraint.

        Args:
            df: DataFrame with scores and rankings
            budget_constraint: Proportion of players to target (0-1)
            score_col: Score column for sorting
            eligible_col: Eligibility column

        Returns:
            DataFrame with 'selected' column added
        """
        df = df.copy()

        # Use config budget if not provided
        if budget_constraint is None:
            budget_constraint = self.business_rules['budget_constraint']

        # Calculate number of players to select
        n_eligible = df[eligible_col].sum()
        n_to_select = int(n_eligible * budget_constraint)

        # Sort by score and select top N
        df['selected'] = False
        eligible_sorted = df[df[eligible_col]].sort_values(score_col, ascending=False)
        top_indices = eligible_sorted.head(n_to_select).index
        df.loc[top_indices, 'selected'] = True

        return df

    def generate_targeting_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary report of targeting decisions.

        Args:
            df: DataFrame with scoring results

        Returns:
            Dictionary with summary statistics
        """
        report = {
            'total_players': len(df),
            'eligible_players': df['eligible'].sum(),
            'selected_players': df['selected'].sum(),
            'selection_rate': df['selected'].sum() / len(df),
            'avg_uplift_selected': df[df['selected']]['uplift_score'].mean(),
            'avg_uplift_not_selected': df[~df['selected']]['uplift_score'].mean(),
            'avg_ltv_selected': df[df['selected']]['ltv_score'].mean(),
            'avg_churn_selected': df[df['selected']]['churn_probability'].mean(),
        }

        # Segment breakdown if available
        if 'segment' in df.columns:
            report['segment_distribution'] = df['segment'].value_counts().to_dict()
            report['selected_by_segment'] = df[df['selected']]['segment'].value_counts().to_dict()

        return report


def score_players(
    df: pd.DataFrame,
    uplift_model,
    ltv_model,
    churn_model,
    feature_engineer,
    config_path: str = 'configs/simulation_config.yaml'
) -> pd.DataFrame:
    """
    Complete scoring pipeline: predict all models and generate final scores.

    Args:
        df: DataFrame with player features
        uplift_model: Fitted uplift model
        ltv_model: Fitted LTV model
        churn_model: Fitted churn model
        feature_engineer: Fitted feature engineer
        config_path: Path to configuration file

    Returns:
        DataFrame with all scores and targeting decisions
    """
    result_df = df.copy()

    # Prepare features
    X = feature_engineer.prepare_features(df, fit_scaler=False)

    # Generate predictions
    result_df['uplift_score'] = uplift_model.predict_uplift(X)
    result_df['ltv_score'] = ltv_model.predict(X)
    result_df['churn_probability'] = churn_model.predict_proba(X)

    # Initialize scoring pipeline
    scorer = ScoringPipeline(config_path)

    # Calculate composite score
    result_df['composite_score'] = scorer.calculate_composite_score(
        result_df['uplift_score'].values,
        result_df['ltv_score'].values,
        result_df['churn_probability'].values
    )

    # Apply business rules
    result_df = scorer.apply_business_rules(result_df)

    # Rank players
    result_df = scorer.rank_players(result_df)

    # Select top players
    result_df = scorer.select_top_players(result_df)

    return result_df


if __name__ == "__main__":
    # Test scoring pipeline
    print("Testing scoring pipeline...")

    from features import FeatureEngineer
    from uplift_model import TLearner
    from ltv_model import LTVModel
    from churn_model import ChurnModel

    # Load test data
    test_df = pd.read_csv('data/player_data_test.csv')

    # Load models
    fe = FeatureEngineer(scale_features=True)
    train_df = pd.read_csv('data/player_data_train.csv')
    fe.prepare_features(train_df, fit_scaler=True)  # Fit scaler

    uplift_model = TLearner.load('data/uplift_model.pkl')
    ltv_model = LTVModel.load('data/ltv_model.pkl')
    churn_model = ChurnModel.load('data/churn_model.pkl')

    # Score players
    print("\nScoring players...")
    scored_df = score_players(test_df, uplift_model, ltv_model, churn_model, fe)

    # Generate report
    scorer = ScoringPipeline()
    report = scorer.generate_targeting_report(scored_df)

    print("\n=== Targeting Report ===")
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    # Show top 10 players
    print("\n=== Top 10 Players by Composite Score ===")
    top_10 = scored_df.nlargest(10, 'composite_score')[
        ['player_id', 'segment', 'uplift_score', 'ltv_score', 'churn_probability', 'composite_score', 'selected']
    ]
    print(top_10.to_string(index=False))

    print("\n[OK] Scoring pipeline test complete!")
