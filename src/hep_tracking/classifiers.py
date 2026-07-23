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
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class RandomForestWrapper(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class XGBoostWrapper(BaseClassifier):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LightGBMWrapper(BaseClassifier):
    def __init__(self, early_stopping_rounds: int = 15, **kwargs):
        self.early_stopping_rounds = early_stopping_rounds
        self.model = lgb.LGBMClassifier(**kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def evaluate_classifier_throughput(
    model, 
    test_dataset: TrackDataset, 
    batch_sizes: tuple[int, ...] = (1, 1000, 10000), 
    num_runs: int = 5
) -> dict[int, float]:
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