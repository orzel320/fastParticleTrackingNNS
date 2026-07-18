import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

def train_random_forest(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Trains a Random Forest classifier.

    :param X_train: Training feature matrix.
    :type X_train: numpy.ndarray
    :param y_train: Training labels.
    :type y_train: numpy.ndarray
    :param X_val: Validation feature matrix (unused by Random Forest, kept for signature consistency).
    :type X_val: numpy.ndarray, optional
    :param y_val: Validation labels.
    :type y_val: numpy.ndarray, optional
    :param kwargs: Additional keyword arguments for RandomForestClassifier.
    :type kwargs: dict
    :return: Trained Random Forest model.
    :rtype: sklearn.ensemble.RandomForestClassifier
    """
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Trains an XGBoost classifier with early stopping if validation data is provided.

    :param X_train: Training feature matrix.
    :type X_train: numpy.ndarray
    :param y_train: Training labels.
    :type y_train: numpy.ndarray
    :param X_val: Validation feature matrix.
    :type X_val: numpy.ndarray, optional
    :param y_val: Validation labels.
    :type y_val: numpy.ndarray, optional
    :param kwargs: Additional keyword arguments for XGBClassifier.
    :type kwargs: dict
    :return: Trained XGBoost model.
    :rtype: xgboost.XGBClassifier
    """
    model = xgb.XGBClassifier(**kwargs)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
        
    return model


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Trains a LightGBM classifier with early stopping if validation data is provided.

    :param X_train: Training feature matrix.
    :type X_train: numpy.ndarray
    :param y_train: Training labels.
    :type y_train: numpy.ndarray
    :param X_val: Validation feature matrix.
    :type X_val: numpy.ndarray, optional
    :param y_val: Validation labels.
    :type y_val: numpy.ndarray, optional
    :param kwargs: Additional keyword arguments for LGBMClassifier.
    :type kwargs: dict
    :return: Trained LightGBM model.
    :rtype: lightgbm.LGBMClassifier
    """
    model = lgb.LGBMClassifier(**kwargs)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
    else:
        model.fit(X_train, y_train)
        
    return model


def evaluate_classifier_throughput(model, X_test, batch_sizes=(1, 1000, 10000), num_runs=5):
    """Evaluates the inference throughput of a trained model across different batch sizes.

    :param model: Trained classification model supporting predict_proba.
    :type model: object
    :param X_test: Test feature matrix to sample batches from.
    :type X_test: numpy.ndarray
    :param batch_sizes: Sequence of batch sizes to evaluate.
    :type batch_sizes: tuple or list
    :param num_runs: Number of timed runs per batch size.
    :type num_runs: int
    :return: Dictionary mapping batch sizes to throughput (samples per second).
    :rtype: dict
    """
    results = {}
    
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

def optimize_hyperparameters(estimator, param_distributions, X_train, y_train, X_val, y_val, n_iter=10, random_state=42):
    """Optimizes hyperparameters using a randomized search over a predefined validation split.

    :param estimator: The scikit-learn compatible classifier instance.
    :type estimator: object
    :param param_distributions: Dictionary with parameters names as keys and distributions or lists of parameters to try.
    :type param_distributions: dict
    :param X_train: Training feature matrix.
    :type X_train: numpy.ndarray
    :param y_train: Training labels.
    :type y_train: numpy.ndarray
    :param X_val: Validation feature matrix.
    :type X_val: numpy.ndarray
    :param y_val: Validation labels.
    :type y_val: numpy.ndarray
    :param n_iter: Number of parameter settings that are sampled.
    :type n_iter: int
    :param random_state: Pseudo random number generator state used for random uniform sampling from lists of possible values instead of scipy.stats distributions.
    :type random_state: int
    :return: Dictionary containing the best parameters found.
    :rtype: dict
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    test_fold = np.concatenate([
        np.full(len(X_train), -1),
        np.zeros(len(X_val))
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