# "fast"matmul
A project I made while learning cpp. This is a python library for "fast" and efficient matrix multiplication (at least compared to base python)

## How is it faster?
1. I use pybind11 and C++ for greater efficiency compared to tortoise-like base python and trivial operations.
2. Optimised algorithms like Strassen's for matrix multiplication, instead of $O(n^3)$ stuff, leading to better $O(n^{log_{2}7})$ time complexity.
3. (Some) CPU parallelisation
4. Power operations: for a fixed size matrix $A$, power operations $A^m$, $m \in \mathbb{N}$ are performed in $O(logm)$ time.
   
## Is it faster?
~100 times faster than completely unoptimised barebones python

## Can it beat NumPy?
Tough luck.

## Installation
1. Requirements C++ 14 or after.
2. Clone and install from the repository.
```
git clone https://github.com/sumhungyee/fastmatmul.git
cd kindafastmatmul
pip install .
```

## Documentation

### Quick Setup
```py
from matmul import Matrix
mat = Matrix([1, 2, 3]) # Vectors are column vectors by default
mat.T() # transposes mat
a = mat.copy() # copies mat
result = a @ Matrix([3, 2, 1]) # Performs a matrix multiplication! Yay!
sum = mat + mat # +, - and * (hadamard product) work, / not defined.
pows = mat ** 10 # matrix mult with itself 10 times, optimised
```

