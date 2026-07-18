import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit


def train_random_forest(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Trenuje klasyfikator Random Forest.

    :param X_train: Macierz cech treningowych.
    :type X_train: numpy.ndarray
    :param y_train: Etykiety treningowe.
    :type y_train: numpy.ndarray
    :param X_val: Macierz cech walidacyjnych (nieużywana przez Random Forest,
        zachowana wyłącznie dla spójności sygnatury z innymi modelami).
    :type X_val: numpy.ndarray, optional
    :param y_val: Etykiety walidacyjne.
    :type y_val: numpy.ndarray, optional
    :param kwargs: Dodatkowe argumenty przekazywane do RandomForestClassifier.
    :type kwargs: dict
    :return: Wytrenowany model Random Forest.
    :rtype: sklearn.ensemble.RandomForestClassifier
    """
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None, **kwargs):
    """Trenuje klasyfikator XGBoost, z wczesnym zatrzymaniem (early stopping),
    jeśli podano dane walidacyjne.

    Uwaga: parametr ``early_stopping_rounds`` przekazywany jest przez ``kwargs``
    bezpośrednio do konstruktora ``XGBClassifier`` — tak wymaga aktualne API XGBoost.

    :param X_train: Macierz cech treningowych.
    :type X_train: numpy.ndarray
    :param y_train: Etykiety treningowe.
    :type y_train: numpy.ndarray
    :param X_val: Macierz cech walidacyjnych.
    :type X_val: numpy.ndarray, optional
    :param y_val: Etykiety walidacyjne.
    :type y_val: numpy.ndarray, optional
    :param kwargs: Dodatkowe argumenty przekazywane do XGBClassifier
        (np. ``early_stopping_rounds``).
    :type kwargs: dict
    :return: Wytrenowany model XGBoost.
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


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=15, **kwargs):
    """Trenuje klasyfikator LightGBM, z wczesnym zatrzymaniem (early stopping),
    jeśli podano dane walidacyjne.

    POPRAWKA: wcześniej ``eval_set`` był przekazywany bez callbacku
    ``lgb.early_stopping``, przez co early stopping w ogóle nie działał —
    model zawsze budował pełną, zadaną liczbę drzew (``n_estimators``),
    nawet jeśli metryka na walidacji przestawała się poprawiać dużo wcześniej.
    To sprawiało, że czas treningu LightGBM był nieporównywalny z XGBoost,
    który early stopping faktycznie wykorzystywał.

    :param X_train: Macierz cech treningowych.
    :type X_train: numpy.ndarray
    :param y_train: Etykiety treningowe.
    :type y_train: numpy.ndarray
    :param X_val: Macierz cech walidacyjnych.
    :type X_val: numpy.ndarray, optional
    :param y_val: Etykiety walidacyjne.
    :type y_val: numpy.ndarray, optional
    :param early_stopping_rounds: Liczba rund bez poprawy metryki walidacyjnej,
        po której trening zostaje przerwany. Używane tylko gdy podano X_val/y_val.
    :type early_stopping_rounds: int
    :param kwargs: Dodatkowe argumenty przekazywane do LGBMClassifier.
    :type kwargs: dict
    :return: Wytrenowany model LightGBM.
    :rtype: lightgbm.LGBMClassifier
    """
    model = lgb.LGBMClassifier(**kwargs)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        )
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_classifier_throughput(model, X_test, batch_sizes=(1, 1000, 10000), num_runs=5):
    """Mierzy przepustowość inferencji wytrenowanego modelu dla różnych rozmiarów wsadu (batch).

    :param model: Wytrenowany model klasyfikacyjny wspierający ``predict_proba``.
    :type model: object
    :param X_test: Macierz cech testowych, z której pobierane są wsady (batche).
    :type X_test: numpy.ndarray
    :param batch_sizes: Sekwencja rozmiarów wsadu do przetestowania.
    :type batch_sizes: tuple or list
    :param num_runs: Liczba pomiarów czasu dla każdego rozmiaru wsadu.
    :type num_runs: int
    :return: Słownik mapujący rozmiar wsadu na przepustowość (próbek/sekundę).
    :rtype: dict
    """
    results = {}

    for batch_size in batch_sizes:
        if batch_size > len(X_test):
            continue

        batch = X_test[:batch_size]

        # rozgrzewka (warm-up) — pomijana w pomiarze czasu
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
    """Optymalizuje hiperparametry metodą losowego przeszukiwania (random search)
    na predefiniowanym podziale walidacyjnym.

    :param estimator: Instancja klasyfikatora zgodna ze scikit-learn.
    :type estimator: object
    :param param_distributions: Słownik z nazwami parametrów jako kluczami
        i listami/rozkładami wartości do przetestowania.
    :type param_distributions: dict
    :param X_train: Macierz cech treningowych.
    :type X_train: numpy.ndarray
    :param y_train: Etykiety treningowe.
    :type y_train: numpy.ndarray
    :param X_val: Macierz cech walidacyjnych.
    :type X_val: numpy.ndarray
    :param y_val: Etykiety walidacyjne.
    :type y_val: numpy.ndarray
    :param n_iter: Liczba losowanych kombinacji parametrów.
    :type n_iter: int
    :param random_state: Ziarno generatora liczb losowych.
    :type random_state: int
    :return: Słownik z najlepszymi znalezionymi parametrami.
    :rtype: dict
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    # -1 oznacza "zawsze w foldzie treningowym", 0 oznacza "jedyny fold walidacyjny"
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