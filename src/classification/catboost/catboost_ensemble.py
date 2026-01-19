import math

import numpy as np
from catboost import CatBoostClassifier


class CatBoost_Ensemble(object):
    # The "Toy Class"
    # https://github.com/yandex-research/GBDT-uncertainty/blob/main/synthetic_classification.ipynb
    def __init__(
        self,
        esize=10,
        iterations=1000,
        lr=0.1,
        random_strength=0,
        border_count=128,
        depth=1,
        seed=100,
        loss_function="logloss",
        bootstrap_type="No",
        verbose=False,
        posterior_sampling=True,
    ):
        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr
        self.random_strength = random_strength
        self.border_count = border_count
        self.ensemble = []
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

    def fit(self, data, eval_set=None):
        for m in self.ensemble:
            m.fit(data[0], y=data[1], eval_set=eval_set)
            print("best iter ", m.get_best_iteration())
            print("best score ", m.get_best_score())

    def predict(self, x):
        probs = []

        for m in self.ensemble:
            prob = m.predict_proba(x)
            probs.append(prob)
        probs = np.stack(probs)
        return probs


def get_grid(ext, resolution=200):
    x = np.linspace(-ext, ext, resolution, dtype=np.float32)
    y = np.linspace(-ext, ext, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, sparse=False)
    return xx, yy


def kl_divergence(probs1, probs2, epsilon=1e-10):
    return np.sum(
        probs1 * (np.log(probs1 + epsilon) - np.log(probs2 + epsilon)), axis=1
    )


def entropy_of_expected(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


def expected_entropy(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)


def mutual_information(probs, epsilon):
    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)
    return eoe - exe


def ensemble_uncertainties(probs: np.ndarray, epsilon: float = 1e-10):
    """
    args:
    probs: shape (esize/no bootstrap iterations, n_samples, n_classes), e.g. (100, 145, 2)
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


def make_new_coordinates(x, y):
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


def eval_ensemble(ens, ext=15, resolution=200):
    """
    args:
    probs: shape (esize, n_samples, n_classes), (20, 40000, 3)
    unks: dictionary
        confidence: shape (n_samples,)
        entropy_of_expected: shape (n_samples,)
        expected_entropy: shape (n_samples,)
        mutual_information: shape (n_samples,)
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
    def __init__(
        self,
        esize=10,
        iterations=1000,
        lr=0.1,
        random_strength=None,
        border_count=None,
        depth=6,
        seed=100,
        load_path=None,
        task_type=None,
        devices=None,
        verbose=True,
        use_base_model=False,
        max_ctr_complexity=None,
        posterior_sampling=True,
        loss_function="Logloss",
    ):
        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr
        self.posterior_sampling = posterior_sampling
        if self.posterior_sampling:
            self.bootstrap_type = "No"
            self.subsample = None
        else:
            self.bootstrap_type = "Bernoulli"
            self.subsample = 0.5
        self.random_strength = random_strength
        self.border_count = border_count
        self.posterior_sampling = posterior_sampling
        self.ensemble = []
        self.use_best_model = use_base_model
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

    def fit(self, data, eval_set=None, save_path="./"):
        for i, m in enumerate(self.ensemble):
            # print(f"TRAINING MODEL {i}\n\n")
            m.fit(data, eval_set=eval_set)
            # m.save_model(f"{save_path}/model{i}.cbm")
            assert np.all(m.classes_ == self.ensemble[0].classes_)

    def save(self, path):
        for i, m in enumerate(self.ensemble):
            m.save_model(f"{path}/model{i}.cmb")

    def predict(self, x):
        probs = []

        for m in self.ensemble:
            assert np.all(m.classes_ == self.ensemble[0].classes_)
            prob = m.predict_proba(x)
            probs.append(prob)

        probs = np.stack(probs)
        return probs


class ClassificationEnsembleSGLB(ClassificationEnsemble):
    def __init__(
        self,
        esize=10,
        iterations=1000,
        lr=0.1,
        random_strength=None,
        border_count=None,
        depth=6,
        seed=100,
        load_path=None,
        task_type=None,
        devices=None,
        verbose=True,
        use_base_model=False,
        max_ctr_complexity=None,
        n_objects=10000,
        loss_function="Logloss",
        colsample_bylevel=None,
        boosting_type="Ordered",
        bootstrap_type="No",
        subsample=None,
        used_ram_limit="12gb",
        min_data_in_leaf=None,
        l2_leaf_reg=None,
        bagging_temperature=None,
    ):
        self.seed = seed
        self.esize = esize
        self.depth = depth
        self.iterations = iterations
        self.lr = lr
        self.random_strength = random_strength
        self.border_count = border_count
        self.ensemble = []
        self.use_best_model = use_base_model
        self.n_objects = n_objects
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

    def fit(self, data, eval_set=None, save_path="./"):
        for i, m in enumerate(self.ensemble):
            # print(f"TRAINING MODEL {i}\n\n")
            m.fit(data, eval_set=eval_set)
            # m.save_model(f"{save_path}/model{i}.cbm")
            assert np.all(m.classes_ == self.ensemble[0].classes_)

    def save(self, path):
        for i, m in enumerate(self.ensemble):
            m.save_model(f"{path}/model{i}.cmb")

    def predict(self, x):
        probs = []

        for m in self.ensemble:
            assert np.all(m.classes_ == self.ensemble[0].classes_)
            prob = m.predict_proba(x)
            probs.append(prob)

        probs = np.stack(probs)
        return probs
