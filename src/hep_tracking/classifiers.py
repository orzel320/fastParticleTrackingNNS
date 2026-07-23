"""Wrapper classes and evaluation utilities for machine learning classifiers."""

import time
from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from hep_tracking.dataset import TrackDataset


class BaseClassifier(ABC):
    """Abstract base class defining the standard interface for track classifiers."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the classifier model.

        Args:
            X_train: Feature matrix for training.
            y_train: Target labels for training.
            X_val: Optional feature matrix for validation (used for early stopping). Defaults to None.
            y_val: Optional target labels for validation. Defaults to None.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the input features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            An array of shape (n_samples, n_classes) containing predicted probabilities.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict deterministic class labels for the input features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            An array of predicted class labels.
        """
        pass


class RandomForestWrapper(BaseClassifier):
    """Wrapper for the scikit-learn RandomForestClassifier."""

    def __init__(self, **kwargs):
        """Initialize the Random Forest model.

        Args:
            **kwargs: Keyword arguments passed directly to the underlying 
                RandomForestClassifier instance.
        """
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the Random Forest model.

        Note that Random Forest does not utilize early stopping natively, so the 
        validation arrays (`X_val` and `y_val`) are ignored if provided.

        Args:
            X_train: Feature matrix for training.
            y_train: Target labels for training.
            X_val: Ignored. Present for interface compatibility.
            y_val: Ignored. Present for interface compatibility.
        """
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            Predicted class probabilities.
        """
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete class labels for the given features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X)


class XGBoostWrapper(BaseClassifier):
    """Wrapper for the XGBoost classifier, supporting validation datasets."""

    def __init__(self, **kwargs):
        """Initialize the XGBoost model.

        Args:
            **kwargs: Keyword arguments passed directly to the underlying 
                XGBClassifier instance.
        """
        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the XGBoost model.

        If validation data is provided, it is passed to the internal model as an 
        evaluation set to track performance.

        Args:
            X_train: Feature matrix for training.
            y_train: Target labels for training.
            X_val: Optional feature matrix for validation.
            y_val: Optional target labels for validation.
        """
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            Predicted class probabilities.
        """
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete class labels for the given features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X)


class LightGBMWrapper(BaseClassifier):
    """Wrapper for the LightGBM classifier, supporting early stopping and validation.
    
    Attributes:
        early_stopping_rounds: Number of rounds to wait for validation metric 
            improvement before terminating training early.
    """

    def __init__(self, early_stopping_rounds: int = 15, **kwargs):
        """Initialize the LightGBM model.

        Args:
            early_stopping_rounds: Consecutive rounds without validation metric 
                improvement needed to trigger early stopping. Defaults to 15.
            **kwargs: Keyword arguments passed directly to the underlying 
                LGBMClassifier instance.
        """
        self.early_stopping_rounds = early_stopping_rounds
        self.model = lgb.LGBMClassifier(**kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """Train the LightGBM model.

        If validation data is provided, the early stopping callback is automatically 
        attached to the training routine to prevent overfitting.

        Args:
            X_train: Feature matrix for training.
            y_train: Target labels for training.
            X_val: Optional feature matrix for validation.
            y_val: Optional target labels for validation.
        """
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_X=X_val,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            Predicted class probabilities.
        """
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete class labels for the given features.

        Args:
            X: Feature matrix to evaluate.

        Returns:
            Predicted class labels.
        """
        return self.model.predict(X)


def evaluate_classifier_throughput(
    model, 
    test_dataset: TrackDataset, 
    batch_sizes: tuple[int, ...] = (1, 1000, 10000), 
    num_runs: int = 5
) -> dict[int, float]:
    """Measure the inference throughput of a model across different batch sizes.

    Executes a warm-up prediction and then measures performance over several iterations 
    to capture the optimal operational speed per batch size.

    Args:
        model: A trained classifier instance implementing `predict_proba`.
        test_dataset: The dataset object containing testing features (`X`).
        batch_sizes: A tuple of integers representing the number of samples 
            to process in a single batch. Defaults to (1, 1000, 10000).
        num_runs: The number of measured timing runs to perform for each batch size. 
            Defaults to 5.

    Returns:
        A dictionary mapping the batch size integer to its calculated throughput 
        in items processed per second.
    """
    results = {}
    X_test = test_dataset.X

    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue

        batch = X_test[:batch_size]
        model.predict_proba(batch)

        execution_times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            model.predict_proba(batch)
            execution_times.append(time.perf_counter() - start_time)

        min_time = min(execution_times)
        results[batch_size] = batch_size / min_time

    return results


def optimize_hyperparameters(
    estimator, 
    param_distributions: dict, 
    train_dataset: TrackDataset, 
    val_dataset: TrackDataset, 
    n_iter: int = 10, 
    random_state: int = 42
) -> dict:
    """Perform a randomized hyperparameter search across a predefined validation split.

    This function securely splits the data by passing an explicit fold array to 
    `PredefinedSplit`. This ensures that `RandomizedSearchCV` evaluates solely on 
    the provided validation dataset without mixing test sets.

    Args:
        estimator: An uninitialized classifier object conforming to the scikit-learn API.
        param_distributions: Dictionary containing parameter names (strings) as keys 
            and lists of parameters or distributions to sample from as values.
        train_dataset: The primary dataset used for fitting the model.
        val_dataset: The holdout dataset used exclusively for scoring each configuration.
        n_iter: The number of parameter settings that are sampled. Defaults to 10.
        random_state: The seed used for reproducible random sampling. Defaults to 42.

    Returns:
        A dictionary containing the parameter settings that yielded the best 
        ROC-AUC score on the validation set.
    """
    X_combined = np.vstack([train_dataset.X, val_dataset.X])
    y_combined = np.concatenate([train_dataset.y, val_dataset.y])

    test_fold = np.concatenate([
        np.full(len(train_dataset), -1),
        np.zeros(len(val_dataset))
    ])
    
    ps = PredefinedSplit(test_fold)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=ps,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X_combined, y_combined)

    return search.best_params_