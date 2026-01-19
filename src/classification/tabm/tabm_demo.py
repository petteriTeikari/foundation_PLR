import numpy as np
import sklearn.datasets


def get_demo_data():
    # Classification.
    n_classes = 2
    assert n_classes >= 2
    # task_type: TaskType = "binclass" if n_classes == 2 else "multiclass"
    X_cont, Y = sklearn.datasets.make_classification(
        n_samples=20000,
        n_features=8,
        n_classes=n_classes,
        n_informative=3,
        n_redundant=2,
    )

    X_cont: np.ndarray = X_cont.astype(np.float32)
    # n_cont_features = X_cont.shape[1]

    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(
        range(n_classes)
    ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

    # >>> Split the dataset.
    all_idx = np.arange(len(Y))
    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.8
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=0.8
    )
    data_numpy = {
        "train": {"x_cont": X_cont[train_idx], "y": Y[train_idx]},
        "val": {"x_cont": X_cont[val_idx], "y": Y[val_idx]},
        "test": {"x_cont": X_cont[test_idx], "y": Y[test_idx]},
    }

    return data_numpy
