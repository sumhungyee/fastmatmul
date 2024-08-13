from timeit import Timer
from matmul import Matrix
import numpy as np
import logging
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
        for r in range(self.rows):
            for c in range(other.cols): # new mat's cols
                for i in range(self.cols):
                    newmat[r][c] = self.mat[r][i] * other.mat[i][c]

        return PyTestMatrix(newmat)



def setup_mats():
    setup_mats = dict()
    
    setup_mats["mat_K1"] = Matrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])
    setup_mats["mat_K2"] = Matrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])

    setup_mats["pymat_K1"] = PyTestMatrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])
    setup_mats["pymat_K2"] = PyTestMatrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])

    e = 12290
    setup_mats["mat_L1"] = Matrix([[random.uniform(-10, 10) for i in range(e)] for j in range(e)])
    setup_mats["mat_L2"] = Matrix([[random.uniform(-10, 10) for i in range(e)] for j in range(e)])

    setup_mats["npymat_L1"] = np.random.rand(e, e)
    setup_mats["npymat_L2"] = np.random.rand(e, e)

    setup_mats["mat_M"] = Matrix([[0 if i != j else 0.5 for i in range(1290)] for j in range(1290)])
    return setup_mats


def benchmark_matmul(mat1, mat2):
    return mat1 @ mat2

if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('benchmarks/benchmark.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    print("Logger set.")

    trials = 10
    m = setup_mats()
    K1, K2, L1, L2, M = m["mat_K1"], m["mat_K2"], m["mat_L1"], m["mat_L2"], m["mat_M"]
    logger.info("cpp bindings:")
    timers = [
        Timer('benchmark_matmul(K1, K2)', 'gc.enable()', globals=globals()),
        Timer('benchmark_matmul(L1, L2)', 'gc.enable()', globals=globals())
    ]
    for ele in timers:
        logger.info(ele.timeit(number=trials) / trials)
    
    PK1, PK2, NPL1, NPL2 = m["pymat_K1"], m["pymat_K2"], m["npymat_L1"], m["npymat_L2"]
    logger.info("python:")
    pytimers = [
        Timer('benchmark_matmul(PK1, PK2)', 'gc.enable()', globals=globals()),
    ]
    for ele in pytimers:
        logger.info(ele.timeit(number=trials) / trials)
    
    logger.info("numpython:")
    numpytimers = [
        Timer('benchmark_matmul(NPL1, NPL2)', 'gc.enable()', globals=globals())
    ]
    for ele in numpytimers:
        logger.info(ele.timeit(number=trials) / trials)
    
    
    
