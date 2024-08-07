import pytest
import random
from matmul import Matrix

@pytest.fixture(scope="function")
def setup_mats():
    setup_mats = dict()
    setup_mats["vec_A"] = Matrix([10, 100, 1000])
    setup_mats["vec_B"] = Matrix([10, 100, 1000])
    setup_mats["mat_A"] = Matrix([(1.2123213123, 12.1231231231), (2, 3)])
    setup_mats["mat_B"] = Matrix(((5, 10), (2, 9)))
    setup_mats["mat_C"] = Matrix(([5, 10], [2, 9]))
    setup_mats["mat_D"] = Matrix([[1, 2], [1, 2]])
    setup_mats["mat_E"] = Matrix(([6, 12], (3, 11)))
    setup_mats["mat_F"] = Matrix([[0, 1], [1, 0]]) # diagonal/symmetric
    setup_mats["mat_I"] = Matrix([[1, 0], (0, 1)]) # identity
    setup_mats["mat_G"] = Matrix([[2, -2], (1, -1)]) # idempotent

    setup_mats["mat_H"] = Matrix([[1 for i in range(128)] for j in range(128)])
    setup_mats["mat_J"] = Matrix([[1 for i in range(129)] for j in range(129)])
    setup_mats["mat_K"] = Matrix([[1 for i in range(1290)] for j in range(1290)])
    setup_mats["mat_L"] = Matrix([[1 for i in range(1220)] for j in range(1290)])
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
    assert -(-(-E)) == -E
    assert D == -(-D)
    assert B + B == 2 * C
    assert 10 * E - 10 * D == 10 * (E - D) == C * 10
    assert F == F.copy().T() == F.copy().T().T()
    assert B != B.copy().T() # B is not symmetric
    assert B == B.copy().T().T() != B.copy().T()
    assert B == (B.copy().T().T().T() * 0.5 + B.copy().T() * 0.5).T()
    assert B.copy().T().T().copy().T().T().copy() == (
        1 + B.copy().T().copy().T().T() * 0.5 + B.copy().T() * 0.5 - 1
        ).T().T().copy().T()
    
def test_insert(setup_mats):
    B = setup_mats["mat_B"]
    C = setup_mats["mat_C"]
    D = setup_mats["mat_D"]
    E = setup_mats["mat_E"]
    new_val = 100
    B[0, 1] = new_val
    assert B.copy().T()[1, 0] == new_val
    assert B.copy().T().T().copy().T()[1, 0] == new_val
    E[0, 1] += new_val
    D[0, 1] += new_val
    assert (E.copy().T() - D.copy().T()).T() == C
    

def test_empty(setup_mats):
    with pytest.raises(RuntimeError, match="Matrix must be nonempty"):
        Matrix([])

    assert Matrix([(1,2), (3,4)]) == Matrix(([1,2], [3,4]))

def test_matmul(setup_mats):
    E = setup_mats["mat_E"]
    I = setup_mats["mat_I"]
    assert I @ I == I
    assert E @ I == I @ E == 1 * E == E


def test_pow(setup_mats):
    D = setup_mats["mat_G"]
    K = D.copy()
    for i in range(2, 90):
        K = K @ D
        assert K == D ** i
    G = setup_mats["mat_G"]
    assert G ** 2 == G == G ** 240

def test_static_methods(setup_mats):
    I = setup_mats["mat_I"]
    assert Matrix.identity(2) == I
    assert Matrix.zeroes(2, 2) == I - I
    assert Matrix.identity(4) != I
    

def test_matmul(setup_mats):
    H = setup_mats["mat_H"]
    
    
    assert (H @ H)[50, 50] == 128
    assert (H @ H)[127, 50] == 128
    assert (H @ H)[126, 127] == 128

def test_matmul_2(setup_mats):
    J = setup_mats["mat_J"]
    L = J @ J
    assert L.dims() == J.dims()
    assert L[127, 127] == 129
    
def test_matmul_large(setup_mats):
    K = setup_mats["mat_K"]
    L = setup_mats["mat_L"]
    J = K @ L
    assert J.dims() == (1290, 1220)
    for i in range(0, 100):
        assert J[i, 0] == 1290
    
