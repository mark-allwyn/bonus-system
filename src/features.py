"""
Feature Engineering Pipeline for Uplift Modeling

Handles feature preparation, validation, and transformation.
Ensures all features are pre-treatment (no data leakage).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Feature engineering for uplift modeling."""

    # Features to use for modeling (exclude labels and IDs)
    FEATURE_COLUMNS = [
        'total_deposits',
        'avg_transaction_size',
        'login_frequency_30d',
        'session_count_30d',
        'days_since_last_login',
        'account_age_days',
        'spending_velocity',
        'sessions_per_login',
        'engagement_score'
    ]

    def __init__(self, scale_features: bool = True):
        """
        Initialize feature engineer.

        Args:
            scale_features: Whether to standardize features
        """
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.feature_names = None

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """
        Prepare feature matrix for modeling.

        Args:
            df: DataFrame with raw features
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            Feature matrix as numpy array
        """
        # Select feature columns
        X = df[self.FEATURE_COLUMNS].copy()

        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()

        # Handle missing values (if any)
        X = X.fillna(X.median())

        # Scale features if requested
        if self.scale_features:
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            return X_scaled
        else:
            return X.values

    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present and valid.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for required columns
        missing_cols = set(self.FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for excessive missing values
        for col in self.FEATURE_COLUMNS:
            if col in df.columns:
                missing_pct = df[col].isna().sum() / len(df) * 100
                if missing_pct > 50:
                    issues.append(f"{col} has {missing_pct:.1f}% missing values")

        # Check for negative values where inappropriate
        non_negative_features = [
            'total_deposits', 'avg_transaction_size', 'login_frequency_30d',
            'session_count_30d', 'days_since_last_login', 'account_age_days'
        ]
        for col in non_negative_features:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"{col} contains negative values")

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for model interpretation."""
        return self.feature_names if self.feature_names else self.FEATURE_COLUMNS


def split_treatment_control(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into treatment and control groups.

    Args:
        df: DataFrame with 'treatment' column

    Returns:
        Tuple of (treatment_df, control_df)
    """
    treatment_df = df[df['treatment'] == 1].copy()
    control_df = df[df['treatment'] == 0].copy()

    return treatment_df, control_df


def create_analysis_splits(df: pd.DataFrame) -> dict:
    """
    Create various data splits for analysis.

    Returns:
        Dictionary with different data splits
    """
    splits = {
        'full': df,
        'treatment': df[df['treatment'] == 1],
        'control': df[df['treatment'] == 0],
    }

    # Split by segment
    for segment in df['segment'].unique():
        splits[f'segment_{segment}'] = df[df['segment'] == segment]

    return splits


def calculate_rolling_features(df: pd.DataFrame, time_windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
    """
    Calculate rolling window features.
    Note: This is a simplified version for demonstration.
    In production, this would require actual time-series data.

    Args:
        df: DataFrame with base features
        time_windows: List of window sizes in days

    Returns:
        DataFrame with additional rolling features
    """
    # For synthetic data, we'll create variations of existing features
    for window in time_windows:
        # Simulate rolling averages with some noise
        df[f'deposits_rolling_{window}d'] = df['total_deposits'] * (1 + np.random.normal(0, 0.1, len(df)))
        df[f'logins_rolling_{window}d'] = df['login_frequency_30d'] * (window / 30) * (1 + np.random.normal(0, 0.15, len(df)))

    return df


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering pipeline...")

    # Load synthetic data
    df = pd.read_csv('data/player_data_train.csv')

    # Initialize feature engineer
    fe = FeatureEngineer(scale_features=True)

    # Validate features
    is_valid, issues = fe.validate_features(df)
    print(f"\nFeature validation: {'[OK] Passed' if is_valid else '[FAILED] Failed'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")

    # Prepare features
    X = fe.prepare_features(df, fit_scaler=True)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature names: {fe.get_feature_importance_names()}")

    # Split treatment/control
    treatment_df, control_df = split_treatment_control(df)
    print(f"\nTreatment group size: {len(treatment_df)}")
    print(f"Control group size: {len(control_df)}")

    print("\n[OK] Feature engineering test complete!")
