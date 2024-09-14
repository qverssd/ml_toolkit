import numpy as np

def normalize(x):
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    return (x - x_min) / (x_max - x_min)

def train_test_split(x, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    indices = np.random.permutation(len(x))
    test_set_size = int(len(x) * test_size)

    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]