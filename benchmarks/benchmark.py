from timeit import Timer
from matmul import Matrix
import gc
import random

def setup_mats():
    setup_mats = dict()
    
    setup_mats["mat_K1"] = Matrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])
    setup_mats["mat_K2"] = Matrix([[random.uniform(-10, 10) for i in range(1290)] for j in range(1290)])

    setup_mats["mat_L1"] = Matrix([[random.uniform(-10, 10) for i in range(4290)] for j in range(4290)])
    setup_mats["mat_L2"] = Matrix([[random.uniform(-10, 10) for i in range(4290)] for j in range(4290)])

    setup_mats["mat_M"] = Matrix([[0 if i != j else 0.5 for i in range(1290)] for j in range(1290)])
    return setup_mats


def benchmark_matmul(mat1, mat2):
    return mat1 @ mat2

if __name__ == "__main__":
    m = setup_mats()
    K1, K2, L1, L2, M = m["mat_K1"], m["mat_K2"], m["mat_L1"], m["mat_L2"], m["mat_M"]
    timers = [
        Timer('benchmark_matmul(K1, K2)', 'gc.enable()', globals=globals()),
        Timer('benchmark_matmul(L1, L2)', 'gc.enable()', globals=globals())
    ]
    for ele in timers:
        print(ele.timeit(number=5))
    
    
    
