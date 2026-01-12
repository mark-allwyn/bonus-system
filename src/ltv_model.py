"""
Lifetime Value (LTV) Prediction Model

Predicts player lifetime value based on behavioral and monetary features.
Used in composite scoring to prioritize high-value players.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


class LTVModel:
    """Lifetime Value prediction model."""

    def __init__(self, random_state: int = 42):
        """
        Initialize LTV model.

        Args:
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LTVModel':
        """
        Fit LTV model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: LTV values (n_samples,)

        Returns:
            Self (fitted model)
        """
        print(f"Training LTV model on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate training metrics
        y_pred = self.model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Training metrics - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LTV values.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            LTV predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate model performance.

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_prediction': y_pred.mean(),
            'std_prediction': y_pred.std()
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
        print(f"LTV model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'LTVModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"LTV model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Test LTV model
    print("Testing LTV model...")

    from features import FeatureEngineer

    # Load data
    train_df = pd.read_csv('data/player_data_train.csv')
    test_df = pd.read_csv('data/player_data_test.csv')

    # Prepare features
    fe = FeatureEngineer(scale_features=True)
    X_train = fe.prepare_features(train_df, fit_scaler=True)
    y_train = train_df['ltv'].values

    X_test = fe.prepare_features(test_df, fit_scaler=False)
    y_test = test_df['ltv'].values

    # Train model
    ltv_model = LTVModel(random_state=42)
    ltv_model.fit(X_train, y_train)

    # Evaluate
    print("\nTest set evaluation:")
    metrics = ltv_model.evaluate(X_test, y_test)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Feature importance
    importance_df = ltv_model.get_feature_importance(fe.get_feature_importance_names())
    print("\nTop 5 important features:")
    print(importance_df.head())

    # Save model
    ltv_model.save('data/ltv_model.pkl')

    print("\n[OK] LTV model test complete!")
