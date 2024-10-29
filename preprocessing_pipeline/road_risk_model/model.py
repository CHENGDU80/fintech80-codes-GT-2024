import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import osmnx as ox
import networkx as nx
import re
from collections import defaultdict
import random
from typing import Dict, Tuple, Optional, List, Union
import warnings
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# ----------------- Define RiskPrediction and RoadRiskModel -----------------


@dataclass
class RiskPrediction:
    """Container for risk prediction results."""
    wait_time: float  # Expected days between accidents
    # Probability of each severity level (1-4)
    severity_probs: Dict[int, float]
    expected_severity: float  # Expected severity value
    risk_score: float  # Raw risk score
    normalized_risk: float  # Risk score normalized to 0-1
    risk_category: str  # Categorical risk level


class RoadRiskModel:
    """
    Advanced model for computing road risk scores combining:
    1. Poisson regression for accident frequency (wait times)
    2. Ordinal logistic regression for severity (1-4 scale)

    Final risk score = expected_severity / wait_time
    """

    RISK_CATEGORIES = {
        0.8: "Very High Risk",
        0.6: "High Risk",
        0.4: "Moderate Risk",
        0.2: "Low Risk",
        0.0: "Very Low Risk"
    }

    SEVERITY_LEVELS = [1, 2, 3, 4]

    def __init__(self, validation_fraction: float = 0.1):
        self.wait_time_model = None
        self.severity_models = None  # Will hold multiple binary logistic models
        self.scaler = StandardScaler()
        self.feature_names = None
        self.validation_fraction = validation_fraction
        self._is_fitted = False

    def _validate_input_data(self, X: pd.DataFrame, severities: Optional[np.ndarray] = None) -> None:
        """Validate input data format and contents."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Features must be provided as a pandas DataFrame")

        if X.empty:
            raise ValueError("Empty feature DataFrame provided")

        if X.isnull().all().any():
            warnings.warn("Some features contain only missing values")

        if severities is not None:
            unique_severities = np.unique(severities)
            if not all(sev in self.SEVERITY_LEVELS for sev in unique_severities):
                raise ValueError(
                    "Severity scores must be integers between 1 and 4")

    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Preprocess features with improved missing value handling and scaling."""
        self._validate_input_data(X)

        if fit:
            self.feature_names = X.columns.tolist()

        X_processed = X.copy()

        # Enhanced missing value handling
        for column in X_processed.columns:
            missing_mask = X_processed[column].isnull()
            if missing_mask.any():
                if X_processed[column].dtype in [np.float64, np.int64]:
                    if fit:
                        self._medians = {column: X_processed[column].median()}
                    X_processed.loc[missing_mask,
                                    column] = self._medians[column]
                else:
                    if fit:
                        self._modes = {
                            column: X_processed[column].mode().iloc[0]}
                    X_processed.loc[missing_mask, column] = self._modes[column]

        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)

        return sm.add_constant(X_scaled)

    def _fit_ordinal_severity_model(self, X: np.ndarray, severities: np.ndarray):
        """
        Fit ordinal logistic regression for severity prediction using multiple binary logistic models.
        Uses the continuation ratio approach.
        """
        self.severity_models = {}

        # Fit a series of binary logistic models
        for level in self.SEVERITY_LEVELS[:-1]:
            # For each level k, model P(Y > k | Y ≥ k)
            mask = severities >= level
            if not np.any(mask):
                continue

            y_binary = (severities > level)[mask]
            X_subset = X[mask]

            model = sm.Logit(y_binary, X_subset)
            self.severity_models[level] = model.fit(disp=0)

    def fit(self, X: pd.DataFrame, wait_times: Union[np.ndarray, List[float]],
            severities: Union[np.ndarray, List[float]]) -> Dict[str, float]:
        """
        Fit both Poisson (wait times) and ordinal logistic (severity) models.
        """
        wait_times = np.array(wait_times)
        severities = np.array(severities)

        self._validate_input_data(X, severities)

        if np.any(wait_times <= 0):
            raise ValueError("Wait times must be positive")

        X_processed = self.preprocess_features(X, fit=True)

        # Split into train/validation
        splits = train_test_split(
            X_processed, wait_times, severities,
            test_size=self.validation_fraction,
            random_state=42
        )
        X_train, X_val, wait_train, wait_val, sev_train, sev_val = splits

        # Fit wait time model
        self.wait_time_model = sm.GLM(
            wait_train,
            X_train,
            family=sm.families.Poisson(link=sm.families.links.log())
        )
        self.wait_time_results = self.wait_time_model.fit()

        # Fit ordinal severity model
        self._fit_ordinal_severity_model(X_train, sev_train)

        self._is_fitted = True

        # Calculate metrics
        metrics = self._calculate_metrics(
            X_train, X_val,
            wait_train, wait_val,
            sev_train, sev_val
        )

        return metrics

    def _predict_severity_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for each severity level using the continuation ratio approach.
        """
        # Initialize probability matrix
        probs = np.zeros((len(X), len(self.SEVERITY_LEVELS)))

        # Calculate conditional probabilities
        prev_mask = np.ones(len(X), dtype=bool)
        cumulative_prob = np.ones(len(X))

        for i, level in enumerate(self.SEVERITY_LEVELS[:-1]):
            if level not in self.severity_models:
                continue

            # P(Y > k | Y ≥ k)
            cond_prob = self.severity_models[level].predict(X[prev_mask])

            # P(Y = k) = P(Y ≥ k) * (1 - P(Y > k | Y ≥ k))
            probs[prev_mask, i] = cumulative_prob[prev_mask] * (1 - cond_prob)

            # Update for next iteration
            cumulative_prob[prev_mask] *= cond_prob
            prev_mask = cumulative_prob > 0.001  # Numerical stability threshold

        # Last level gets remaining probability
        probs[prev_mask, -1] = cumulative_prob[prev_mask]

        return probs

    def _calculate_metrics(self, X_train, X_val, wait_train, wait_val,
                           sev_train, sev_val) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics."""
        wait_pred_train = self.wait_time_results.predict(X_train)
        wait_pred_val = self.wait_time_results.predict(X_val)

        sev_probs_train = self._predict_severity_probs(X_train)
        sev_probs_val = self._predict_severity_probs(X_val)

        # Calculate expected severity values
        exp_sev_train = np.sum(
            sev_probs_train * np.array(self.SEVERITY_LEVELS), axis=1)
        exp_sev_val = np.sum(
            sev_probs_val * np.array(self.SEVERITY_LEVELS), axis=1)

        return {
            'wait_time_aic': self.wait_time_results.aic,
            'wait_time_rmse_train': np.sqrt(np.mean((wait_train - wait_pred_train) ** 2)),
            'wait_time_rmse_val': np.sqrt(np.mean((wait_val - wait_pred_val) ** 2)),
            'severity_rmse_train': np.sqrt(np.mean((sev_train - exp_sev_train) ** 2)),
            'severity_rmse_val': np.sqrt(np.mean((sev_val - exp_sev_val) ** 2)),
            'severity_accuracy_train': np.mean(np.argmax(sev_probs_train, axis=1) + 1 == sev_train),
            'severity_accuracy_val': np.mean(np.argmax(sev_probs_val, axis=1) + 1 == sev_val)
        }

    def _categorize_risk(self, normalized_score: float) -> str:
        """Convert normalized risk score to categorical risk level."""
        for threshold, category in self.RISK_CATEGORIES.items():
            if normalized_score >= threshold:
                return category
        return self.RISK_CATEGORIES[0.0]

    def predict_risk(self, X: pd.DataFrame) -> List[RiskPrediction]:
        """
        Predict comprehensive risk metrics for road segments.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_processed = self.preprocess_features(X, fit=False)

        # Predict wait times
        wait_times = self.wait_time_results.predict(X_processed)

        # Predict severity probabilities
        severity_probs_matrix = self._predict_severity_probs(X_processed)

        # Calculate expected severity values
        expected_severities = np.sum(
            severity_probs_matrix * np.array(self.SEVERITY_LEVELS), axis=1)

        # Compute risk scores
        risk_scores = expected_severities / wait_times

        # Normalize scores
        normalized_scores = (risk_scores - risk_scores.min()) / (
            risk_scores.max() - risk_scores.min()
        )

        # Create prediction objects
        predictions = []
        for i in range(len(X)):
            severity_probs = dict(
                zip(self.SEVERITY_LEVELS, severity_probs_matrix[i]))
            pred = RiskPrediction(
                wait_time=wait_times[i],
                severity_probs=severity_probs,
                expected_severity=expected_severities[i],
                risk_score=risk_scores[i],
                normalized_risk=normalized_scores[i],
                risk_category=self._categorize_risk(normalized_scores[i])
            )
            predictions.append(pred)

        return predictions