import scipy.sparse as sp

from lightfm.model_selection import train_test_split


def test_train_test_split():
    intrct = sp.rand(2000, 1000, density=0.01, random_state=42)
    train, test = train_test_split(intrct, train_size=0.8, random_state=69)
    assert (intrct - train - test).nnz == 0
    assert train.nnz == intrct.nnz * 0.80
