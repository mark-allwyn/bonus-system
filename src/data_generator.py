"""
Synthetic Data Generator for Uplift Modeling Paper

Generates realistic player data with four distinct segments:
- Persuadables: Engage only with bonus (positive uplift)
- Sure Things: Engage regardless of bonus (no uplift)
- Lost Causes: Never engage (no uplift)
- Sleeping Dogs: Disengage with bonus (negative uplift)
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, Tuple


class PlayerDataGenerator:
    """Generate synthetic player data for uplift modeling experiments."""

    def __init__(self, config_path: str = 'configs/simulation_config.yaml'):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.random_state = self.config['experiment']['random_seed']
        np.random.seed(self.random_state)

    def generate_players(self) -> pd.DataFrame:
        """
        Generate complete synthetic player dataset.

        Returns:
            DataFrame with player features, segment labels, and treatment assignment
        """
        n = self.config['population']['total_size']
        segments = self.config['population']['segments']

        # Assign segments based on configured proportions
        segment_labels = self._assign_segments(n, segments)

        # Generate features for each segment
        features = self._generate_features(segment_labels)

        # Assign treatment (randomized)
        treatment = self._assign_treatment(n)

        # Generate outcomes based on segment and treatment
        outcomes = self._generate_outcomes(segment_labels, treatment)

        # Combine into dataframe
        df = pd.DataFrame({
            'player_id': range(1, n + 1),
            'segment': segment_labels,
            'treatment': treatment,
            'outcome': outcomes,
            **features
        })

        return df

    def _assign_segments(self, n: int, segments: Dict[str, float]) -> np.ndarray:
        """Assign players to segments based on configured proportions."""
        segment_names = list(segments.keys())
        segment_probs = list(segments.values())

        # Validate proportions sum to 1.0
        assert abs(sum(segment_probs) - 1.0) < 0.001, "Segment proportions must sum to 1.0"

        return np.random.choice(segment_names, size=n, p=segment_probs)

    def _generate_features(self, segments: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate player features based on segment characteristics."""
        n = len(segments)
        features = {}
        feature_config = self.config['features']

        # Generate each feature based on segment-specific distributions
        for feature_name, segment_params in feature_config.items():
            feature_values = np.zeros(n)

            for segment in segment_params.keys():
                mask = segments == segment
                mean, std = segment_params[segment]

                # Generate from normal distribution, ensure non-negative
                values = np.random.normal(mean, std, mask.sum())
                values = np.maximum(values, 0)  # No negative values

                feature_values[mask] = values

            features[feature_name] = feature_values

        # Add derived features
        features['spending_velocity'] = features['total_deposits'] / np.maximum(features['account_age_days'], 1)
        features['sessions_per_login'] = features['session_count_30d'] / np.maximum(features['login_frequency_30d'], 1)
        features['engagement_score'] = (
            features['login_frequency_30d'] * 0.3 +
            features['session_count_30d'] * 0.3 +
            (30 - np.minimum(features['days_since_last_login'], 30)) * 0.4
        )

        return features

    def _assign_treatment(self, n: int) -> np.ndarray:
        """Randomly assign treatment (1) or control (0)."""
        treatment_prob = self.config['experiment']['treatment_probability']
        return np.random.binomial(1, treatment_prob, n)

    def _generate_outcomes(self, segments: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """
        Generate binary outcomes based on segment and treatment.

        Outcome model:
        P(Y=1 | segment, treatment) = baseline_rate + treatment_effect * treatment
        """
        n = len(segments)
        outcomes = np.zeros(n, dtype=int)

        baseline_config = self.config['baseline_engagement']
        treatment_effects = self.config['treatment_effects']

        for segment_name in baseline_config.keys():
            mask = segments == segment_name

            # Calculate engagement probability for this segment
            baseline = baseline_config[segment_name]
            effect = treatment_effects[segment_name]

            # Control group: baseline rate
            control_mask = mask & (treatment == 0)
            control_probs = np.full(control_mask.sum(), baseline)
            outcomes[control_mask] = np.random.binomial(1, control_probs)

            # Treatment group: baseline + treatment effect
            treatment_mask = mask & (treatment == 1)
            treatment_probs = np.clip(baseline + effect, 0, 1)
            outcomes[treatment_mask] = np.random.binomial(1, treatment_probs)

        return outcomes

    def calculate_true_uplift(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the true uplift for each player based on their segment.
        This is the ground truth for evaluation purposes.
        """
        uplift_map = self.config['treatment_effects']
        df['true_uplift'] = df['segment'].map(uplift_map)
        return df

    def generate_ltv_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lifetime value labels for LTV model training.
        LTV correlates with spending behavior and engagement.
        """
        # LTV is a function of deposits, engagement, and some noise
        ltv = (
            df['total_deposits'] * 2.5 +
            df['engagement_score'] * 50 +
            df['account_age_days'] * 1.5 +
            np.random.normal(0, 200, len(df))
        )
        df['ltv'] = np.maximum(ltv, 0)
        return df

    def generate_churn_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn probability labels for churn model training.
        Churn is inversely related to engagement.
        """
        # Higher days since last login and lower engagement = higher churn risk
        churn_score = (
            df['days_since_last_login'] * 0.04 +
            (30 - df['login_frequency_30d']) * 0.02 +
            np.random.normal(0, 0.1, len(df))
        )
        df['churn_probability'] = np.clip(churn_score, 0, 1)
        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV."""
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"\nSegment distribution:\n{df['segment'].value_counts()}")
        print(f"\nTreatment distribution:\n{df['treatment'].value_counts()}")
        print(f"\nOutcome distribution:\n{df['outcome'].value_counts()}")


def main():
    """Generate and save synthetic datasets."""
    generator = PlayerDataGenerator()

    # Generate full dataset
    print("Generating synthetic player data...")
    df = generator.generate_players()

    # Add ground truth uplift
    df = generator.calculate_true_uplift(df)

    # Add LTV and churn labels
    df = generator.generate_ltv_labels(df)
    df = generator.generate_churn_labels(df)

    # Save full dataset
    generator.save_dataset(df, 'data/player_data.csv')

    # Create train/test split (70/30)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['segment'])

    generator.save_dataset(train_df, 'data/player_data_train.csv')
    generator.save_dataset(test_df, 'data/player_data_test.csv')

    print("\n[OK] Data generation complete!")


if __name__ == "__main__":
    main()
