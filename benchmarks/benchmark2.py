from matmul import *
import random
from timeit import Timer
import numpy as np
import logging
import gc

def setup_mats(e):
    setup_mats = dict()
    
    setup_mats["mat_L1"] = Matrix([[random.uniform(-10, 10) for i in range(e)] for j in range(e)])
    setup_mats["mat_L2"] = Matrix([[random.uniform(-10, 10) for i in range(e)] for j in range(e)])

    setup_mats["npymat_L1"] = np.random.rand(e, e)
    setup_mats["npymat_L2"] = np.random.rand(e, e)

    return setup_mats


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('benchmarks/benchmark.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    print("Logger set.")
    e = int(input())
    trials = int(input())
    m = setup_mats(e)

    A, B = m["mat_L1"], m["mat_L2"]
    nA, nB = m["npymat_L1"], m["npymat_L2"]

    timers = [
        Timer('A @ B', 'gc.enable()', globals=globals()),
        Timer('nA @ nB', 'gc.enable()', globals=globals())
    ]
    logger.info(f"cpp bindings and numpy, {e}")
    for ele in timers:
        logger.info(ele.timeit(number=trials) / trials)