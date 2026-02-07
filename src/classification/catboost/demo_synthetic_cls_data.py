import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from src.classification.catboost.catboost_ensemble import (
    CatBoost_Ensemble,
    eval_ensemble,
    make_new_coordinates,
)

sns.set()


def create_single_spiral(n_points, angle_offset, noise=0.1):
    """
    Generate points along a single Archimedean spiral.

    Parameters
    ----------
    n_points : int
        Number of points to generate along the spiral.
    angle_offset : float
        Angular offset in radians for the spiral starting position.
    noise : float, optional
        Noise level for random perturbation of points. Default is 0.1.

    Returns
    -------
    ndarray
        Array of shape (n_points, 2) containing x and y coordinates.
    """
    # Create numbers in the range [0., 6 pi], where the initial square root maps the uniformly
    # distributed points to lie mainly towards the upper limit of the range
    n = np.sqrt(np.random.rand(n_points, 1)) * 3 * (2 * np.pi)

    # Calculate the x and y coordinates of the spiral and add random noise to each coordinate
    x = -np.cos(n + angle_offset) * n**2 + np.random.randn(
        n_points, 1
    ) * noise * n * np.sqrt(n)
    y = np.sin(n + angle_offset) * n**2 + np.random.randn(
        n_points, 1
    ) * noise * n * np.sqrt(n)

    return np.hstack((x, y))


def create_spirals(n_points, n_spirals=3, noise=0.1, seed=100):
    """
    Generate a multi-spiral dataset for classification.

    Parameters
    ----------
    n_points : int
        Number of points per spiral.
    n_spirals : int, optional
        Number of spiral classes to generate. Default is 3.
    noise : float, optional
        Noise level for random perturbation of points. Default is 0.1.
    seed : int, optional
        Random seed for reproducibility. Default is 100.

    Returns
    -------
    X : ndarray
        Feature array of shape (n_points * n_spirals, 2) with x, y coordinates.
    Y : ndarray
        Label array of shape (n_points * n_spirals,) with class indices.
    """
    np.random.seed(seed)

    angle_separation = 2 * np.pi / n_spirals  # The angle separation between each spiral

    X, Y = [], []
    for i in range(n_spirals):
        X.append(
            create_single_spiral(
                n_points, angle_offset=angle_separation * i, noise=noise
            )
        )
        Y.append(np.ones(n_points) * i)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.longlong)


class SpiralDataset(Dataset):
    """The Toy Three Class Dataset."""

    def __init__(self, size, noise, scale, seed=100):
        """
        Initialize the spiral dataset.

        Parameters
        ----------
        size : int
            Number of points per spiral class.
        noise : float
            Noise level for random perturbation of spiral points.
        scale : float
            Scale factor for the dataset (stored but not applied to data).
        seed : int, optional
            Random seed for reproducibility. Default is 100.
        """
        self.scale = scale
        self.size = size
        self.noise = noise
        self.seed = seed

        self.x, self.y = create_spirals(
            n_points=size, n_spirals=3, noise=noise, seed=seed
        )
        return

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples (size * 3 for three spiral classes).
        """
        return self.size * 3

    def __getitem__(self, idx):
        """
        Get a sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Tuple of (features, label) where features is a 2D coordinate array.
        """
        return self.x[idx], self.y[idx]

    def plot(self, ax=None, s=10.0, alpha=0.7):
        """
        Plot the spiral dataset with different colors per class.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates a new figure.
        s : float, optional
            Marker size for scatter plot. Default is 10.0.
        alpha : float, optional
            Transparency of markers. Default is 0.7.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plotted data.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set(aspect="equal")
            colors = sns.color_palette()  # (3, start=0.2, rot=-0.7, light=0.75)
            for i in range(3):
                plt.scatter(
                    *np.hsplit(self.x[self.y == i], 2),
                    color=colors[i],
                    s=s,
                    alpha=alpha,
                )
        plt.ylim(-400, 400)
        plt.xlim(-400, 400)

        return ax


def create_spiral_train_and_eval_data():
    """
    Create training and evaluation datasets from spiral data with derived features.

    Generates two spiral datasets with different random seeds and transforms
    the 2D coordinates into expanded feature representations.

    Returns
    -------
    new_data : tuple
        Training data as (features, labels) with shape ((4500, 9), (4500,)).
    eval_data : tuple
        Evaluation data as (features, labels) with shape ((4500, 9), (4500,)).
    """
    # data = SpiralDataset(size=1500, scale=3, noise=0.4)
    # data.plot()

    data = create_spirals(1500, noise=0.4, seed=24)
    new_features = []
    for i in range(len(data[0])):
        element = data[0][i]
        x = element[0]
        y = element[1]
        new_coordinates = make_new_coordinates(x, y)
        new_features.append(new_coordinates)
    # tuple: (features, labels) (4500,9), (4500,)
    new_data = (np.array(new_features), data[1])
    # print(new_data)

    data = create_spirals(1500, noise=0.4, seed=51)
    new_features = []
    for i in range(len(data[0])):
        element = data[0][i]
        x = element[0]
        y = element[1]
        new_coordinates = make_new_coordinates(x, y)
        new_features.append(new_coordinates)
    # tuple: (features, labels) (4500,9), (4500,)
    eval_data = (np.array(new_features), data[1])

    return new_data, eval_data


def demo_spiral_ensemble():
    """
    Demonstrate CatBoost ensemble classification on spiral dataset.

    Trains a CatBoost ensemble on synthetic spiral data and evaluates
    prediction probabilities and uncertainties over a 2D grid.

    Returns
    -------
    None
        Prints ensemble predictions and uncertainties.
    """
    new_data, eval_data = create_spiral_train_and_eval_data()
    ens = CatBoost_Ensemble(
        esize=20,
        iterations=1000,
        lr=0.1,
        depth=6,
        seed=2,
        random_strength=100,
        loss_function="MultiClass",
    )
    ens.fit(new_data, eval_set=eval_data)
    # probs (20, 40000, 3)
    # unks: dictionary
    probs, unks = eval_ensemble(ens, ext=600, resolution=200)
