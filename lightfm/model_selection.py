from __future__ import print_function

import numbers
from math import ceil, floor

import numpy as np

import scipy.sparse as sp


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    COPYRIGHT: Scikit-learn 0.18 (sklearn/utils/validation.py)
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _validate_shuffle_split_init(test_size, train_size):
    """Validation helper to check the test_size and train_size at init
    NOTE This does not take into account the number of samples which is known
    only at split

    COPYRIGHT: Scikit-learn 0.18 (sklearn/model_selection/_split.py)
    """
    if test_size is None and train_size is None:
        raise ValueError('test_size and train_size can not both be None')

    if test_size is not None:
        if np.asarray(test_size).dtype.kind == 'f':
            if test_size >= 1.:
                raise ValueError(
                    'test_size=%f should be smaller '
                    'than 1.0 or be an integer' % test_size)
        elif np.asarray(test_size).dtype.kind != 'i':
            # int values are checked during split based on the input
            raise ValueError("Invalid value for test_size: %r" % test_size)

    if train_size is not None:
        if np.asarray(train_size).dtype.kind == 'f':
            if train_size >= 1.:
                raise ValueError("train_size=%f should be smaller "
                                 "than 1.0 or be an integer" % train_size)
            elif (np.asarray(test_size).dtype.kind == 'f' and
                    (train_size + test_size) > 1.):
                raise ValueError('The sum of test_size and train_size = %f, '
                                 'should be smaller than 1.0. Reduce '
                                 'test_size and/or train_size.' %
                                 (train_size + test_size))
        elif np.asarray(train_size).dtype.kind != 'i':
            # int values are checked during split based on the input
            raise ValueError("Invalid value for train_size: %r" % train_size)


def _validate_shuffle_split(n_samples, test_size, train_size):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)

    COPYRIGHT: Scikit-learn 0.18 (sklearn/model_selection/_split.py)
    """
    if (test_size is not None and np.asarray(test_size).dtype.kind == 'i' and
            test_size >= n_samples):
        raise ValueError('test_size=%d should be smaller than the number of '
                         'samples %d' % (test_size, n_samples))

    if (train_size is not None and np.asarray(train_size).dtype.kind == 'i' and
            train_size >= n_samples):
        raise ValueError("train_size=%d should be smaller than the number of"
                         " samples %d" % (train_size, n_samples))

    if np.asarray(test_size).dtype.kind == 'f':
        n_test = ceil(test_size * n_samples)
    elif np.asarray(test_size).dtype.kind == 'i':
        n_test = float(test_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif np.asarray(train_size).dtype.kind == 'f':
        n_train = floor(train_size * n_samples)
    else:
        n_train = float(train_size)

    if test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d. Reduce test_size and/or '
                         'train_size.' % (n_train + n_test, n_samples))

    return int(n_train), int(n_test)


def train_test_split(interactions, test_size=None, train_size=0.75,
                     random_state=None):
    """Split arrays or matrices into random train and test subsets
    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.
    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.
    stratify : array-like or None (default is None)
        If not None, data is split in a stratified fashion, using this as
        the groups array.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    ...
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_train
    [2, 0, 3]
    >>> X_test
    array([[2, 3],
           [8, 9]])
    >>> y_test
    [1, 4]
    """
    n_samples = interactions.nnz
    _validate_shuffle_split_init(test_size, train_size)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size)
    rng = check_random_state(random_state)
    idx = rng.choice(interactions.nnz, size=n_train, replace=False)
    train_mask = np.zeros(interactions.nnz, dtype=bool)
    train_mask[idx] = True
    train_mat = sp.coo_matrix(
        (interactions.data[train_mask],
         (interactions.row[train_mask], interactions.col[train_mask])),
        shape=interactions.shape)
    test_mat = sp.coo_matrix(
        (interactions.data[~train_mask],
         (interactions.row[~train_mask], interactions.col[~train_mask])),
        shape=interactions.shape)
    return train_mat, test_mat
