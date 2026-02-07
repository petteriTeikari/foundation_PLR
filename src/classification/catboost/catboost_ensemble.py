import math
from typing import Optional, Union

import numpy as np
from catboost import CatBoostClassifier, Pool


class CatBoost_Ensemble(object):
    """
    Toy CatBoost ensemble for uncertainty estimation.

    A simple ensemble of CatBoost classifiers for synthetic classification tasks.
    Based on: https://github.com/yandex-research/GBDT-uncertainty/blob/main/synthetic_classification.ipynb

    Parameters
    ----------
    esize : int, optional
        Number of models in the ensemble. Default is 10.
    iterations : int, optional
        Number of boosting iterations per model. Default is 1000.
    lr : float, optional
        Learning rate. Default is 0.1.
    random_strength : float, optional
        Random strength for scoring splits. Default is 0.
    border_count : int, optional
        Number of splits for numerical features. Default is 128.
    depth : int, optional
        Depth of the tree. Default is 1.
    seed : int, optional
        Random seed for reproducibility. Default is 100.
    loss_function : str, optional
        Loss function to optimize. Default is "logloss".
    bootstrap_type : str, optional
        Bootstrap type. Default is "No".
    verbose : bool, optional
        Whether to print training progress. Default is False.
    posterior_sampling : bool, optional
        Whether to use posterior sampling. Default is True.
    """

    def __init__(
        self,
        esize: int = 10,
        iterations: int = 1000,
        lr: float = 0.1,
        random_strength: float = 0,
        border_count: int = 128,
        depth: int = 1,
        seed: int = 100,
        loss_function: str = "logloss",
        bootstrap_type: str = "No",
        verbose: bool = False,
        posterior_sampling: bool = True,
    ) -> None:
        self.seed: int = seed
        self.esize: int = esize
        self.depth: int = depth
        self.iterations: int = iterations
        self.lr: float = lr
        self.random_strength: float = random_strength
        self.border_count: int = border_count
        self.ensemble: list[CatBoostClassifier] = []
        for e in range(self.esize):
            model = CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.lr,
                border_count=self.border_count,
                random_strength=self.random_strength,
                loss_function=loss_function,
                verbose=verbose,
                bootstrap_type=bootstrap_type,
                posterior_sampling=posterior_sampling,
                random_seed=self.seed + e,
            )
            self.ensemble.append(model)

    def fit(
        self,
        data: tuple[np.ndarray, np.ndarray],
        eval_set: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """
        Fit all models in the ensemble.

        Parameters
        ----------
        data : tuple
            Tuple of (X, y) where X is the feature matrix and y is the target.
        eval_set : tuple, optional
            Evaluation dataset for early stopping. Default is None.

        Returns
        -------
        None
        """
        for m in self.ensemble:
            m.fit(data[0], y=data[1], eval_set=eval_set)
            print("best iter ", m.get_best_iteration())
            print("best score ", m.get_best_score())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Get predictions from all ensemble members.

        Parameters
        ----------
        x : array-like
            Input features to predict on.

        Returns
        -------
        np.ndarray
            Stacked probability predictions with shape (esize, n_samples, n_classes).
        """
        probs: list[np.ndarray] = []

        for m in self.ensemble:
            prob = m.predict_proba(x)
            probs.append(prob)
        stacked_probs = np.stack(probs)
        return stacked_probs


def get_grid(ext: float, resolution: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D grid for visualization.

    Parameters
    ----------
    ext : float
        Extent of the grid in each direction (from -ext to +ext).
    resolution : int, optional
        Number of points along each axis. Default is 200.

    Returns
    -------
    tuple of np.ndarray
        Tuple (xx, yy) of meshgrid arrays, each with shape (resolution, resolution).
    """
    x = np.linspace(-ext, ext, resolution, dtype=np.float32)
    y = np.linspace(-ext, ext, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, sparse=False)
    return xx, yy


def kl_divergence(
    probs1: np.ndarray, probs2: np.ndarray, epsilon: float = 1e-10
) -> np.ndarray:
    """
    Compute KL divergence between two probability distributions.

    Parameters
    ----------
    probs1 : np.ndarray
        First probability distribution with shape (n_samples, n_classes).
    probs2 : np.ndarray
        Second probability distribution with shape (n_samples, n_classes).
    epsilon : float, optional
        Small constant to avoid log(0). Default is 1e-10.

    Returns
    -------
    np.ndarray
        KL divergence values with shape (n_samples,).
    """
    return np.sum(
        probs1 * (np.log(probs1 + epsilon) - np.log(probs2 + epsilon)), axis=1
    )


def entropy_of_expected(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute entropy of the expected (mean) probability distribution.

    Parameters
    ----------
    probs : np.ndarray
        Probability predictions with shape (esize, n_samples, n_classes).
    epsilon : float, optional
        Small constant to avoid log(0). Default is 1e-10.

    Returns
    -------
    np.ndarray
        Entropy values with shape (n_samples,).
    """
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


def expected_entropy(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute the expected entropy across ensemble members.

    Parameters
    ----------
    probs : np.ndarray
        Probability predictions with shape (esize, n_samples, n_classes).
    epsilon : float, optional
        Small constant to avoid log(0). Default is 1e-10.

    Returns
    -------
    np.ndarray
        Expected entropy values with shape (n_samples,).
    """
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)


def mutual_information(probs: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compute mutual information as the difference between entropy measures.

    Mutual information quantifies epistemic uncertainty (model uncertainty).

    Parameters
    ----------
    probs : np.ndarray
        Probability predictions with shape (esize, n_samples, n_classes).
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray
        Mutual information values with shape (n_samples,).
    """
    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    return eoe - exe


def ensemble_uncertainties(
    probs: np.ndarray, epsilon: float = 1e-10
) -> dict[str, np.ndarray]:
    """
    Compute multiple uncertainty measures from ensemble predictions.

    Parameters
    ----------
    probs : np.ndarray
        Probability predictions with shape (esize, n_samples, n_classes).
        For example, (100, 145, 2) for 100 bootstrap iterations, 145 samples, 2 classes.
    epsilon : float, optional
        Small constant to avoid log(0). Default is 1e-10.

    Returns
    -------
    dict
        Dictionary containing:
        - 'confidence': Maximum mean probability per sample (n_samples,).
        - 'entropy_of_expected': Entropy of mean predictions (n_samples,).
        - 'expected_entropy': Mean entropy across ensemble (n_samples,).
        - 'mutual_information': Epistemic uncertainty (n_samples,).
    """
    assert len(probs.shape) == 3, "Input must be a 3D array, not {}D array".format(
        len(probs.shape)
    )
    mean_probs = np.mean(probs, axis=0)  # (n_samples, n_classes)
    conf = np.max(mean_probs, axis=1)  # (n_samples,)

    eoe = entropy_of_expected(probs, epsilon)  # (n_samples,)
    exe = expected_entropy(probs, epsilon)  # (n_samples,)
    mutual_info = eoe - exe  # (n_samples,)

    uncertainty = {
        "confidence": conf,
        "entropy_of_expected": eoe,
        "expected_entropy": exe,
        "mutual_information": mutual_info,
    }

    return uncertainty


def make_new_coordinates(x: float, y: float) -> list[float]:
    """
    Generate extended feature coordinates from 2D input.

    Creates 9 features from 2D coordinates including linear combinations
    and Euclidean distance.

    Parameters
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.

    Returns
    -------
    list
        List of 9 features: [x, y, x+y, x-y, 2x+y, x-2y, x+2y, 2x-y, sqrt(x^2+y^2)].
    """
    return [
        x,
        y,
        x + y,
        x - y,
        2 * x + y,
        x - 2 * y,
        x + 2 * y,
        2 * x - y,
        math.sqrt(x * x + y * y),
    ]


def eval_ensemble(
    ens: Union["CatBoost_Ensemble", "ClassificationEnsemble"],
    ext: float = 15,
    resolution: int = 200,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Evaluate ensemble predictions and uncertainties over a 2D grid.

    Parameters
    ----------
    ens : CatBoost_Ensemble or ClassificationEnsemble
        Trained ensemble model with a predict method.
    ext : float, optional
        Extent of the grid in each direction. Default is 15.
    resolution : int, optional
        Number of points along each axis. Default is 200.

    Returns
    -------
    probs : np.ndarray
        Probability predictions with shape (esize, n_samples, n_classes).
        For resolution=200, n_samples=40000.
    unks : dict
        Dictionary of uncertainty measures, each with shape (n_samples,):
        - 'confidence': Maximum mean probability.
        - 'entropy_of_expected': Entropy of mean predictions.
        - 'expected_entropy': Mean entropy across ensemble.
        - 'mutual_information': Epistemic uncertainty.
    """
    xx, yy = get_grid(ext, resolution)  # (200, 200)
    # inputs = np.stack((xx.ravel(), yy.ravel()), axis=1)
    inputs_ext = np.array(
        [
            make_new_coordinates(x, y)
            for x, y in np.stack((xx.ravel(), yy.ravel()), axis=1)
        ]
    )  # (40000, 9)
    probs = ens.predict(inputs_ext)
    unks = ensemble_uncertainties(probs)

    return probs, unks


# https://raw.githubusercontent.com/yandex-research/GBDT-uncertainty/refs/heads/main/gbdt_uncertainty/ensemble.py
class ClassificationEnsemble(object):
    """
    Production-ready CatBoost ensemble for classification with uncertainty.

    A more configurable ensemble implementation supporting GPU training,
    model persistence, and posterior sampling for uncertainty estimation.
    Based on: https://github.com/yandex-research/GBDT-uncertainty

    Parameters
    ----------
    esize : int, optional
        Number of models in the ensemble. Default is 10.
    iterations : int, optional
        Number of boosting iterations per model. Default is 1000.
    lr : float, optional
        Learning rate. Default is 0.1.
    random_strength : float, optional
        Random strength for scoring splits. Default is None.
    border_count : int, optional
        Number of splits for numerical features. Default is None.
    depth : int, optional
        Depth of the tree. Default is 6.
    seed : int, optional
        Random seed for reproducibility. Default is 100.
    load_path : str, optional
        Path to load pre-trained models from. Default is None.
    task_type : str, optional
        CatBoost task type ('CPU' or 'GPU'). Default is None.
    devices : str, optional
        GPU device IDs. Default is None.
    verbose : bool, optional
        Whether to print training progress. Default is True.
    use_base_model : bool, optional
        Whether to use best model (early stopping). Default is False.
    max_ctr_complexity : int, optional
        Maximum CTR complexity. Default is None.
    posterior_sampling : bool, optional
        Whether to use posterior sampling. Default is True.
    loss_function : str, optional
        Loss function to optimize. Default is "Logloss".
    """

    def __init__(
        self,
        esize: int = 10,
        iterations: int = 1000,
        lr: float = 0.1,
        random_strength: Optional[float] = None,
        border_count: Optional[int] = None,
        depth: int = 6,
        seed: int = 100,
        load_path: Optional[str] = None,
        task_type: Optional[str] = None,
        devices: Optional[str] = None,
        verbose: bool = True,
        use_base_model: bool = False,
        max_ctr_complexity: Optional[int] = None,
        posterior_sampling: bool = True,
        loss_function: str = "Logloss",
    ) -> None:
        self.seed: int = seed
        self.esize: int = esize
        self.depth: int = depth
        self.iterations: int = iterations
        self.lr: float = lr
        self.posterior_sampling: bool = posterior_sampling
        if self.posterior_sampling:
            self.bootstrap_type: str = "No"
            self.subsample: Optional[float] = None
        else:
            self.bootstrap_type = "Bernoulli"
            self.subsample = 0.5
        self.random_strength: Optional[float] = random_strength
        self.border_count: Optional[int] = border_count
        self.posterior_sampling = posterior_sampling
        self.ensemble: list[CatBoostClassifier] = []
        self.use_best_model: bool = use_base_model
        for e in range(self.esize):
            model = CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.lr,
                border_count=self.border_count,
                random_strength=self.random_strength,
                loss_function=loss_function,
                verbose=verbose,
                posterior_sampling=self.posterior_sampling,
                use_best_model=self.use_best_model,
                max_ctr_complexity=max_ctr_complexity,
                random_seed=self.seed + e,
                subsample=self.subsample,
                task_type=task_type,
                bootstrap_type=self.bootstrap_type,
                devices=devices,
            )
            self.ensemble.append(model)

        if load_path is not None:
            for i, m in enumerate(self.ensemble):
                m.load_model(f"{load_path}/model{i}.cbm")

    def fit(
        self,
        data: Union[Pool, tuple[np.ndarray, np.ndarray]],
        eval_set: Optional[Union[Pool, tuple[np.ndarray, np.ndarray]]] = None,
        save_path: str = "./",
    ) -> None:
        """
        Fit all models in the ensemble.

        Parameters
        ----------
        data : CatBoost Pool or tuple
            Training data as Pool object or (X, y) tuple.
        eval_set : Pool or tuple, optional
            Evaluation dataset for monitoring. Default is None.
        save_path : str, optional
            Path for saving models (currently unused). Default is "./".

        Returns
        -------
        None
        """
        for i, m in enumerate(self.ensemble):
            # print(f"TRAINING MODEL {i}\n\n")
            m.fit(data, eval_set=eval_set)
            # m.save_model(f"{save_path}/model{i}.cbm")
            assert np.all(m.classes_ == self.ensemble[0].classes_)

    def save(self, path: str) -> None:
        """
        Save all ensemble models to disk.

        Parameters
        ----------
        path : str
            Directory path where models will be saved as model0.cmb, model1.cmb, etc.

        Returns
        -------
        None
        """
        for i, m in enumerate(self.ensemble):
            m.save_model(f"{path}/model{i}.cmb")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Get predictions from all ensemble members.

        Parameters
        ----------
        x : array-like
            Input features to predict on.

        Returns
        -------
        np.ndarray
            Stacked probability predictions with shape (esize, n_samples, n_classes).
        """
        probs: list[np.ndarray] = []

        for m in self.ensemble:
            assert np.all(m.classes_ == self.ensemble[0].classes_)
            prob = m.predict_proba(x)
            probs.append(prob)

        stacked_probs = np.stack(probs)
        return stacked_probs


class ClassificationEnsembleSGLB(ClassificationEnsemble):
    """
    CatBoost ensemble using Stochastic Gradient Langevin Boosting (SGLB).

    Uses Langevin dynamics for posterior sampling, providing better uncertainty
    estimates than standard ensembles. Inherits from ClassificationEnsemble.

    Parameters
    ----------
    esize : int, optional
        Number of models in the ensemble. Default is 10.
    iterations : int, optional
        Number of boosting iterations per model. Default is 1000.
    lr : float, optional
        Learning rate. Default is 0.1.
    random_strength : float, optional
        Random strength for scoring splits. Default is None.
    border_count : int, optional
        Number of splits for numerical features. Default is None.
    depth : int, optional
        Depth of the tree. Default is 6.
    seed : int, optional
        Random seed for reproducibility. Default is 100.
    load_path : str, optional
        Path to load pre-trained models from. Default is None.
    task_type : str, optional
        CatBoost task type ('CPU' or 'GPU'). Default is None.
    devices : str, optional
        GPU device IDs. Default is None.
    verbose : bool, optional
        Whether to print training progress. Default is True.
    use_base_model : bool, optional
        Whether to use best model (early stopping). Default is False.
    max_ctr_complexity : int, optional
        Maximum CTR complexity. Default is None.
    n_objects : int, optional
        Number of objects for diffusion temperature scaling. Default is 10000.
    loss_function : str, optional
        Loss function to optimize. Default is "Logloss".
    colsample_bylevel : float, optional
        Column sampling ratio per level. Default is None.
    boosting_type : str, optional
        Boosting type ('Ordered' or 'Plain'). Default is "Ordered".
    bootstrap_type : str, optional
        Bootstrap type. Default is "No".
    subsample : float, optional
        Subsample ratio of training instances. Default is None.
    used_ram_limit : str, optional
        RAM limit for CatBoost. Default is "12gb".
    _min_data_in_leaf : int, optional
        Minimum samples in leaf (unused). Default is None.
    _l2_leaf_reg : float, optional
        L2 regularization (unused). Default is None.
    bagging_temperature : float, optional
        Bagging temperature for Bayesian bootstrap. Default is None.
    """

    def __init__(
        self,
        esize: int = 10,
        iterations: int = 1000,
        lr: float = 0.1,
        random_strength: Optional[float] = None,
        border_count: Optional[int] = None,
        depth: int = 6,
        seed: int = 100,
        load_path: Optional[str] = None,
        task_type: Optional[str] = None,
        devices: Optional[str] = None,
        verbose: bool = True,
        use_base_model: bool = False,
        max_ctr_complexity: Optional[int] = None,
        n_objects: int = 10000,
        loss_function: str = "Logloss",
        colsample_bylevel: Optional[float] = None,
        boosting_type: str = "Ordered",
        bootstrap_type: str = "No",
        subsample: Optional[float] = None,
        used_ram_limit: str = "12gb",
        _min_data_in_leaf: Optional[int] = None,
        _l2_leaf_reg: Optional[float] = None,
        bagging_temperature: Optional[float] = None,
    ) -> None:
        self.seed: int = seed
        self.esize: int = esize
        self.depth: int = depth
        self.iterations: int = iterations
        self.lr: float = lr
        self.random_strength: Optional[float] = random_strength
        self.border_count: Optional[int] = border_count
        self.ensemble: list[CatBoostClassifier] = []
        self.use_best_model: bool = use_base_model
        self.n_objects: int = n_objects
        for e in range(self.esize):
            model = CatBoostClassifier(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.lr,
                border_count=self.border_count,
                random_strength=self.random_strength,
                loss_function=loss_function,
                verbose=verbose,
                langevin=True,
                diffusion_temperature=self.n_objects,
                # model_shrink_rate=0.5 / self.n_objects,
                use_best_model=self.use_best_model,
                max_ctr_complexity=max_ctr_complexity,
                random_seed=self.seed + e,
                subsample=subsample,
                task_type=task_type,
                bootstrap_type=bootstrap_type,
                devices=devices,
                colsample_bylevel=colsample_bylevel,
                boosting_type=boosting_type,
                used_ram_limit=used_ram_limit,
                bagging_temperature=bagging_temperature,
            )
            self.ensemble.append(model)

        # if load_path is not None:
        #     for i, m in enumerate(self.ensemble):
        #         m.load_model(f"{load_path}/model{i}.cbm")

    def fit(
        self,
        data: Union[Pool, tuple[np.ndarray, np.ndarray]],
        eval_set: Optional[Union[Pool, tuple[np.ndarray, np.ndarray]]] = None,
        save_path: str = "./",
    ) -> None:
        """
        Fit all SGLB models in the ensemble.

        Parameters
        ----------
        data : CatBoost Pool or tuple
            Training data as Pool object or (X, y) tuple.
        eval_set : Pool or tuple, optional
            Evaluation dataset for monitoring. Default is None.
        save_path : str, optional
            Path for saving models (currently unused). Default is "./".

        Returns
        -------
        None
        """
        for i, m in enumerate(self.ensemble):
            # print(f"TRAINING MODEL {i}\n\n")
            m.fit(data, eval_set=eval_set)
            # m.save_model(f"{save_path}/model{i}.cbm")
            assert np.all(m.classes_ == self.ensemble[0].classes_)

    def save(self, path: str) -> None:
        """
        Save all SGLB ensemble models to disk.

        Parameters
        ----------
        path : str
            Directory path where models will be saved as model0.cmb, model1.cmb, etc.

        Returns
        -------
        None
        """
        for i, m in enumerate(self.ensemble):
            m.save_model(f"{path}/model{i}.cmb")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Get predictions from all SGLB ensemble members.

        Parameters
        ----------
        x : array-like
            Input features to predict on.

        Returns
        -------
        np.ndarray
            Stacked probability predictions with shape (esize, n_samples, n_classes).
        """
        probs: list[np.ndarray] = []

        for m in self.ensemble:
            assert np.all(m.classes_ == self.ensemble[0].classes_)
            prob = m.predict_proba(x)
            probs.append(prob)

        stacked_probs = np.stack(probs)
        return stacked_probs
