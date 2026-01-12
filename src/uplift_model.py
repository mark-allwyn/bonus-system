"""
Uplift Modeling Implementation

Implements T-Learner (Two-Model) approach for causal inference:
- Trains separate models for treatment and control groups
- Predicts uplift as difference in predicted probabilities
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Tuple, Optional
import pickle


class TLearner:
    """
    T-Learner (Two-Model Learner) for uplift modeling.

    Trains separate models for treatment and control groups to estimate
    Individual Treatment Effect (ITE):
    ITE(x) = P(Y=1|X=x,T=1) - P(Y=1|X=x,T=0)
    """

    def __init__(self, base_model=None, random_state: int = 42):
        """
        Initialize T-Learner.

        Args:
            base_model: Sklearn-compatible classifier (default: RandomForestClassifier)
            random_state: Random seed for reproducibility
        """
        if base_model is None:
            self.treatment_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=50,
                random_state=random_state,
                n_jobs=-1
            )
            self.control_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=50,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            # Clone the base model for both groups
            from sklearn.base import clone
            self.treatment_model = clone(base_model)
            self.control_model = clone(base_model)

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> 'TLearner':
        """
        Fit separate models for treatment and control groups.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary outcome (n_samples,)
            treatment: Treatment indicator (n_samples,)

        Returns:
            Self (fitted model)
        """
        # Split into treatment and control
        treatment_mask = treatment == 1
        control_mask = treatment == 0

        X_treatment = X[treatment_mask]
        y_treatment = y[treatment_mask]

        X_control = X[control_mask]
        y_control = y[control_mask]

        print(f"Training treatment model on {len(X_treatment)} samples...")
        self.treatment_model.fit(X_treatment, y_treatment)

        print(f"Training control model on {len(X_control)} samples...")
        self.control_model.fit(X_control, y_control)

        self.is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """
        Predict uplift (treatment effect) for each sample.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Uplift predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get probability predictions from both models
        prob_treatment = self.treatment_model.predict_proba(X)[:, 1]
        prob_control = self.control_model.predict_proba(X)[:, 1]

        # Uplift is the difference
        uplift = prob_treatment - prob_control

        return uplift

    def predict_response(self, X: np.ndarray, treatment_group: str = 'treatment') -> np.ndarray:
        """
        Predict response probability for a specific group.

        Args:
            X: Feature matrix
            treatment_group: 'treatment' or 'control'

        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if treatment_group == 'treatment':
            return self.treatment_model.predict_proba(X)[:, 1]
        elif treatment_group == 'control':
            return self.control_model.predict_proba(X)[:, 1]
        else:
            raise ValueError("treatment_group must be 'treatment' or 'control'")

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance from both models.

        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        treatment_importance = self.treatment_model.feature_importances_
        control_importance = self.control_model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else [f'feature_{i}' for i in range(len(treatment_importance))],
            'treatment_importance': treatment_importance,
            'control_importance': control_importance,
            'avg_importance': (treatment_importance + control_importance) / 2
        })

        return importance_df.sort_values('avg_importance', ascending=False)

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'TLearner':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model


class ResponseModel:
    """
    Traditional response model (for comparison with uplift).

    Trains a single model on treatment group only to predict who
    will respond. This is the standard approach that can perform
    poorly by targeting Sure Things and Sleeping Dogs.
    """

    def __init__(self, base_model=None, random_state: int = 42):
        """Initialize response model."""
        if base_model is None:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=50,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.model = base_model

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> 'ResponseModel':
        """
        Fit model on treatment group only.

        Args:
            X: Feature matrix
            y: Binary outcome
            treatment: Treatment indicator (only treatment group is used)

        Returns:
            Self (fitted model)
        """
        # Use only treatment group
        treatment_mask = treatment == 1
        X_treatment = X[treatment_mask]
        y_treatment = y[treatment_mask]

        print(f"Training response model on {len(X_treatment)} treatment samples...")
        self.model.fit(X_treatment, y_treatment)

        self.is_fitted = True
        return self

    def predict_response(self, X: np.ndarray) -> np.ndarray:
        """Predict response probability."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Response model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ResponseModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Response model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Test uplift model
    print("Testing T-Learner uplift model...")

    from features import FeatureEngineer

    # Load data
    train_df = pd.read_csv('data/player_data_train.csv')
    test_df = pd.read_csv('data/player_data_test.csv')

    # Prepare features
    fe = FeatureEngineer(scale_features=True)
    X_train = fe.prepare_features(train_df, fit_scaler=True)
    y_train = train_df['outcome'].values
    treatment_train = train_df['treatment'].values

    X_test = fe.prepare_features(test_df, fit_scaler=False)
    y_test = test_df['outcome'].values
    treatment_test = test_df['treatment'].values

    # Train T-Learner
    print("\n=== Training T-Learner ===")
    t_learner = TLearner(random_state=42)
    t_learner.fit(X_train, y_train, treatment_train)

    # Predict uplift
    uplift_pred = t_learner.predict_uplift(X_test)
    print(f"\nUplift predictions - Mean: {uplift_pred.mean():.4f}, Std: {uplift_pred.std():.4f}")
    print(f"Uplift range: [{uplift_pred.min():.4f}, {uplift_pred.max():.4f}]")

    # Feature importance
    importance_df = t_learner.get_feature_importance(fe.get_feature_importance_names())
    print("\nTop 5 important features:")
    print(importance_df.head())

    # Save model
    t_learner.save('data/uplift_model.pkl')

    # Train response model for comparison
    print("\n=== Training Response Model ===")
    response_model = ResponseModel(random_state=42)
    response_model.fit(X_train, y_train, treatment_train)

    response_pred = response_model.predict_response(X_test)
    print(f"Response predictions - Mean: {response_pred.mean():.4f}, Std: {response_pred.std():.4f}")

    response_model.save('data/response_model.pkl')

    print("\n[OK] Uplift model test complete!")
