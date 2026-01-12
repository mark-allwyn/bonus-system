"""
Churn Probability Prediction Model

Predicts likelihood of player churn based on engagement patterns.
Used in composite scoring to prioritize at-risk players.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import pickle


class ChurnModel:
    """Churn probability prediction model."""

    def __init__(self, random_state: int = 42):
        """
        Initialize churn model.

        Args:
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle potential class imbalance
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ChurnModel':
        """
        Fit churn model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Churn labels - binary or probabilities (n_samples,)

        Returns:
            Self (fitted model)
        """
        # Convert probabilities to binary if needed
        if y.max() <= 1.0 and y.min() >= 0.0:
            # Assume these are probabilities, convert to binary for training
            y_binary = (y > 0.5).astype(int)
        else:
            y_binary = y.astype(int)

        print(f"Training churn model on {len(X)} samples...")
        print(f"Churn rate: {y_binary.mean():.2%}")

        self.model.fit(X, y_binary)
        self.is_fitted = True

        # Calculate training metrics
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_binary, y_pred_proba)
        acc = accuracy_score(y_binary, (y_pred_proba > 0.5).astype(int))

        print(f"Training metrics - AUC: {auc:.3f}, Accuracy: {acc:.3f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict churn probability.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Churn probabilities (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary churn labels.

        Args:
            X: Feature matrix
            threshold: Probability threshold for classification

        Returns:
            Binary predictions (n_samples,)
        """
        probas = self.predict_proba(X)
        return (probas > threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y_true: True churn labels (binary or probabilities)

        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to binary if needed
        if y_true.max() <= 1.0:
            y_binary = (y_true > 0.5).astype(int)
        else:
            y_binary = y_true.astype(int)

        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(y_binary, y_pred_proba),
            'accuracy': accuracy_score(y_binary, y_pred),
            'mean_pred_proba': y_pred_proba.mean(),
            'std_pred_proba': y_pred_proba.std(),
            'churn_rate_true': y_binary.mean(),
            'churn_rate_pred': y_pred.mean()
        }

        return metrics

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance = self.model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names if feature_names else [f'feature_{i}' for i in range(len(importance))],
            'importance': importance
        })

        return importance_df.sort_values('importance', ascending=False)

    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Churn model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ChurnModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Churn model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Test churn model
    print("Testing churn model...")

    from features import FeatureEngineer

    # Load data
    train_df = pd.read_csv('data/player_data_train.csv')
    test_df = pd.read_csv('data/player_data_test.csv')

    # Prepare features
    fe = FeatureEngineer(scale_features=True)
    X_train = fe.prepare_features(train_df, fit_scaler=True)
    y_train = train_df['churn_probability'].values

    X_test = fe.prepare_features(test_df, fit_scaler=False)
    y_test = test_df['churn_probability'].values

    # Train model
    churn_model = ChurnModel(random_state=42)
    churn_model.fit(X_train, y_train)

    # Evaluate
    print("\nTest set evaluation:")
    metrics = churn_model.evaluate(X_test, y_test)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Feature importance
    importance_df = churn_model.get_feature_importance(fe.get_feature_importance_names())
    print("\nTop 5 important features:")
    print(importance_df.head())

    # Save model
    churn_model.save('data/churn_model.pkl')

    print("\n[OK] Churn model test complete!")
