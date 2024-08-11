# "fast"matmul
A project I made while learning cpp. This is a python library for "fast" and efficient matrix multiplication (at least compared to base python)

## How is it faster?
1. I use pybind11 and C++ for greater efficiency compared to tortoise-like base python and trivial operations.
2. Optimised algorithms like Strassen's for matrix multiplication, instead of $O(n^3)$ stuff, leading to better $O(n^{log_{2}7})$ time complexity.
3. (Some) CPU parallelisation
4. Power operations: for a fixed size matrix $A$, power operations $A^m$, $m \in \mathbb{N}$ are performed in $O(logm)$ time.
5. Optimised padding for strassen's. Instead of padding to the smallest power of 2, iteratively find an integer slightly larger than half of itself until that integer is smaller than the threshold, then multiply back.
   - i.e. With `#define LARGEMATRIXFORSTRASSEN 64` as the threshold, instead of:
   - ```cpp
     static size_t get_2n(size_t length) {
            size_t power = 1;
            while (power < length) {
               power <<= 1;
            }
            return power;
     }
     ```
   - Do:
   - ```cpp
     static size_t get_2n(size_t length) {
            size_t count = 0;
            while (length > LARGEMATRIXFORSTRASSEN) {
                length >>= 1;
                length += 1;
                count += 1;
            }
            return length << count;
     }
     ```
   
## Is it faster?
~300 times faster than completely unoptimised barebones python for semi-large (1000 x 1000) matrices

## Can it beat NumPy?
At large enough sizes, yes! >:) This size probably depends on your computer architecture.

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

