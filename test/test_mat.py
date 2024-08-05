import pytest
import random
from matmul import Matrix

@pytest.fixture(scope="function")
def setup_mats():
    setup_mats = dict()
    setup_mats["vec_A"] = Matrix([10, 100, 1000])
    setup_mats["vec_B"] = Matrix([10, 100, 1000])
    setup_mats["mat_A"] = Matrix([[1.2123213123, 12.1231231231], [2, 3]])
    setup_mats["mat_B"] = Matrix([[5, 10], [1, 9]])
    setup_mats["mat_C"] = Matrix([[5, 10], [1, 9]])
    setup_mats["mat_D"] = Matrix([[1, 2], [1, 2]])
    setup_mats["mat_E"] = Matrix([[6, 12], [2, 11]])
    setup_mats["mat_F"] = Matrix([[0, 1], [1, 0]]) # diagonal/symmetric
    yield setup_mats

def test_eq(setup_mats):
    assert setup_mats["mat_B"] == setup_mats["mat_C"]
    assert setup_mats["vec_A"] == setup_mats["vec_B"] == setup_mats["vec_A"]
    assert Matrix([1]) == Matrix([1])

def test_transpose(setup_mats):
    A = setup_mats["mat_A"].copy()
    B = setup_mats["mat_A"].copy()
    assert A.T().T().T().T() == B
    # A is modified
    assert A.T()[0, 1] == B[1, 0]

def test_sum(setup_mats):
    B = setup_mats["mat_B"]
    C = setup_mats["mat_C"]
    D = setup_mats["mat_D"]
    E = setup_mats["mat_E"]
    F = setup_mats["mat_F"]
    assert (C + D) == E
    assert E - D == C
    assert B + B == 2 * C
    assert F == F.copy().T() == F.copy().T().T()
    assert B != B.copy().T() # B is not symmetric
    assert B == B.copy().T().T() != B.copy().T()
    assert B == (B.copy().T().T().T() * 0.5 + B.copy().T() * 0.5).T()
    
    
