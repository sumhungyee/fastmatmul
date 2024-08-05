import pytest
import random
from matmul import Matrix

@pytest.fixture(scope="module")
def setup_mats():
    mats = dict()
    mats["vec_A"] = Matrix([10, 100, 1000])
    mats["vec_B"] = Matrix([10, 100, 1000])
    mats["mat_A"] = Matrix([[1.2123213123, 12.123123123123123123], [2, 3]])
    mats["mat_B"] = Matrix([[5, 10], [1, 9]])
    yield mats

def test_eq(mats):
    assert mats["mat_B"] == Matrix([[5, 10], [1, 9]])
    assert mats["vec_A"] == mats["vec_B"]
    assert Matrix([1]) == Matrix([1])

def test_transpose(mats):
    A = mats["mat_A"].copy()
    B = mats["mat_A"].copy()
    assert A.T().T().T().T() == B
    # A is modified
    assert A.T()[0, 1] == B[1, 0]