import pytest
import random
from matmul import Matrix

@pytest.fixture(scope="module")
def setup_mats():
    mats = dict()
    mats["vec_A"] = Matrix([10, 100, 1000])
    mats["vec_B"] = Matrix([10, 100, 1000])
    mats["mat_A"] = Matrix([[
        1.2123213123, 12.123123123123123123
    ], [
        2, 3
    ]])