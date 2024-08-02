#include <iostream>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;

class Matrix {
    public:
        unique_ptr<double[]> mat;
        size_t rows, cols;
    
    Matrix(const size_t rows, const size_t cols) : rows(rows), cols(cols) {
        if (rows <= 0 || 0 >= cols) {
            throw std::out_of_range("Matrix dimensions must be positive");
        }
        mat = make_unique<double[]>(rows * cols);
    }

    // copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        mat = make_unique<double[]>(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) mat[i] = other.mat[i];
    }

    Matrix(const py::list& list) {
        size_t py_rows = list.size();
        if (list.empty()) {
            throw std::runtime_error("Matrix must be nonempty");
        } 
            
        size_t py_cols;
        const py::list item = list.attr("__getitem__")(0);
        // assume its a list     
        py_cols = item.size();

        this->rows = py_rows;
        this->cols = py_cols;
        this->mat = make_unique<double[]>(rows * cols);
        // copy first row in
        for (size_t i = 0; i < py_cols; ++i) {
            this->mat[i] = item.attr("__getitem__")(i).cast<double>();
        }

        for (size_t i = 1; i < py_rows; ++i) {
            const py::list item = list.attr("__getitem__")(i);
            
            if (item.size() != py_cols) {
                throw std::runtime_error("Matrix rows have different lengths");
            }
            // copy values in row by row
            for (size_t j = 0; j < py_cols; ++j) {
                this->mat[i * py_cols + j] = item.attr("__getitem__")(j).cast<double>();
            }
          
        }
    }

    // copy assignmnt
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            this->rows = other.rows;
            this->cols = other.cols;
            this->mat = make_unique<double[]>(other.rows * other.cols);
            for (size_t i = 0; i < rows * cols; ++i) this->mat[i] = other.mat[i];
            return *this;
        }
        return *this;
    }

    double get_item(size_t r, size_t c) const {

        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        return mat[r * cols + c];
    }

    Matrix copy() {
        return Matrix(*this);
    }

};

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(matmul, m) {
    m.doc() = "A fun module I built while learning cpp, wip"; // still in the works
    m.def("add", &add, "A function that adds two numbers");
    //https://stackoverflow.com/questions/49301317/pybind11-passing-a-python-list-to-c-style-array
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<const py::list&>())
        .def("assign", &Matrix::operator=) //https://stackoverflow.com/questions/60745723/pybind11-wrapping-overloaded-assignment-operator
        .def("copy", &Matrix::copy);
}