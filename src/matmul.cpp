#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;

class Matrix {
    public:
        double* mat;
        size_t rows, cols;
    
    Matrix(const size_t rows, const size_t cols) : rows(rows), cols(cols) {
        if (rows <= 0 || 0 >= cols) {
            throw std::out_of_range("Matrix dimensions must be positive");
        }
        mat = new double[rows * cols];
    }

    Matrix(const py::list& list) {
        if (list.empty()) {
            throw std::out_of_range("Matrix cannot be empty!");
        }
    }

    // copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        mat = new double[rows * cols];
        for (size_t i = 0; i < rows * cols; ++i) mat[i] = other.mat[i];
    }

    // copy assignmnt
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] this->mat;
            this->rows = other.rows;
            this->cols = other.cols;
            this->mat = new double[other.rows * other.cols];
            for (size_t i = 0; i < rows * cols; ++i) this->mat[i] = other.mat[i];
            return *this;
        }
        return *this;
    }

    ~Matrix() {
        delete[] mat;
    }

    double get_item(size_t r, size_t c) const {

        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return mat[r * cols + c];
    }

};

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(matmul, m) {
    m.doc() = "A fun module I built while learning cpp, wip"; // still in the works
    m.def("add", &add, "A function that adds two numbers");
}