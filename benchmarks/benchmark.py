from timeit import Timer
from matmul import Matrix
import gc
import random

class PyTestMatrix:
    def __init__(self, lsofls: list[list[float]]) -> None:
        self.mat = list(lsofls)
        curr_len = len(lsofls[0])
        for item in lsofls:
            assert len(item) == curr_len
        self.rows = len(lsofls)
        self.cols = curr_len
    
    def __matmul__(self, other):
        assert self.cols == other.rows
        newmat = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        for r in self.rows:
            for c in other.cols: # new mat's cols
                for i in range(self.cols):
                    newmat[r][c] = self.mat[r][i] * other.mat[i][c]

        return PyTestMatrix(newmat)



def setup_mats():
    setup_mats = dict()
    
    setup_mats["mat_K1"] = Matrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])
    setup_mats["mat_K2"] = Matrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])

    setup_mats["pymat_K1"] = PyTestMatrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])
    setup_mats["pymat_K2"] = PyTestMatrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])

    setup_mats["mat_L1"] = Matrix([[random.uniform(-10, 10) for i in range(4290)] for j in range(4290)])
    setup_mats["mat_L2"] = Matrix([[random.uniform(-10, 10) for i in range(4290)] for j in range(4290)])

    setup_mats["pymat_L1"] = PyTestMatrix([[random.uniform(-10, 10) for i in range(4290)] for j in range(4290)])
    setup_mats["pymat_L2"] = PyTestMatrix([[random.uniform(-10, 10) for i in range(4290)] for j in range(4290)])

    setup_mats["mat_M"] = Matrix([[0 if i != j else 0.5 for i in range(1290)] for j in range(1290)])
    return setup_mats


def benchmark_matmul(mat1, mat2):
    return mat1 @ mat2

if __name__ == "__main__":
    trials = 5
    m = setup_mats()
    K1, K2, L1, L2, M = m["mat_K1"], m["mat_K2"], m["mat_L1"], m["mat_L2"], m["mat_M"]
    print("cpp bindings:")
    timers = [
        Timer('benchmark_matmul(K1, K2)', 'gc.enable()', globals=globals()),
        Timer('benchmark_matmul(L1, L2)', 'gc.enable()', globals=globals())
    ]
    for ele in timers:
        print(ele.timeit(number=trials))
    
    PK1, PK2, PL1, PL2 = m["pymat_K1"], m["pymat_K2"], m["pymat_L1"], m["pymat_L2"]
    print("python:")
    pytimers = [
        Timer('benchmark_matmul(PK1, PK2)', 'gc.enable()', globals=globals()),
        Timer('benchmark_matmul(PL1, PL2)', 'gc.enable()', globals=globals())
    ]

    for ele in pytimers:
        print(ele.timeit(number=trials))
    
    
    
