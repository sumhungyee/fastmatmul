#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define DECIMALPLACES 1000000
#define LARGEMATRIX 53
#define SMALL 3

namespace py = pybind11;

using namespace std;

class Matrix {
    private:
        unique_ptr<double[]> mat;
        bool data_is_transposed = false;
    
    public:
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
        
        if (other.is_transposed()) {
            // transposing already swaps this->rows with this->cols. no need to swap rows and cols
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat[j + i * cols] = other.mat[i + j * other.rows];
                }
            }

        } else {
            for (size_t i = 0; i < rows * cols; ++i) mat[i] = other.mat[i];
        }
        
    }

    Matrix(const py::list& list) {
        size_t py_rows = list.size();
        if (list.empty()) {
            throw std::runtime_error("Matrix must be nonempty");
        } else if (!py::isinstance<py::list>(list.attr("__getitem__")(0))) {
            auto outer= py::list();
            outer.append(list);
            *this = Matrix(outer);
            return;
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

    bool is_transposed() const {
        return this->data_is_transposed;
    }

    std::vector<double> get_array() const {
        size_t size = rows * cols;
        return std::vector<double>(this->mat.get(), this->mat.get() + size);
    }

    // copy assignmnt
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            this->rows = other.rows;
            this->cols = other.cols;
            this->data_is_transposed = other.is_transposed();
            this->mat = std::make_unique<double[]>(other.rows * other.cols);
            for (size_t i = 0; i < rows * cols; ++i) this->mat[i] = other.mat[i];
            return *this;
        }
        return *this;
    }

    double get_item(std::tuple<const size_t, const size_t> tup) const {
        size_t r = std::get<0>(tup);
        size_t c = std::get<1>(tup);
        
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }

        if (this->is_transposed()) {
            return mat[c * rows + r];
        } else {
            return mat[r * cols + c];
        }
        
    }

    Matrix copy() {
        return Matrix(*this);
    }

    string repr() {
        string repr_str = "";
        bool rows_too_big = rows > LARGEMATRIX;
        bool cols_too_big = cols > LARGEMATRIX;
        std::vector<size_t> rows_arr;
        std::vector<size_t> cols_arr;
        std::vector<size_t> temp_arr;
        // rows or cols too big, show only a few
        if (rows_too_big) {
            rows_arr.reserve(SMALL * 2);
            rows_arr = {0, 1, 2, rows-3, rows-2, rows-1};
        } else {
            for (size_t i = 0; i < rows; ++i) rows_arr.push_back(i);
        }

        if (cols_too_big) {
            cols_arr.reserve(SMALL * 2);
            cols_arr = {0, 1, 2, cols-3, cols-2, cols-1};
        } else {
            for (size_t i = 0; i < cols; ++i) cols_arr.push_back(i);
        }

        // representation
        for (const size_t r : rows_arr) {
            repr_str += "[";
            for (const size_t c : cols_arr) {
                repr_str += std::to_string(std::round(get_item(std::make_tuple(r, c)) * DECIMALPLACES) / DECIMALPLACES);
                if (c < cols - 1) {
                    repr_str += ", ";
                }
                if (cols_too_big && c == SMALL - 1) {
                    repr_str += "...";
                }
            }
            repr_str += "]\n";
            if (rows_too_big && r == SMALL - 1) {
                repr_str += "...\n";
            }
        }

        // works but eh too long
        
        /*
        if (rows < LARGEMATRIX && cols < LARGEMATRIX) {
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    repr_str += std::to_string(std::round(get_item(r, c) * DECIMALPLACES) / DECIMALPLACES);
                    if (c < cols - 1) {
                        repr_str += ", ";
                    }
                }
                repr_str += "\n";
            }

        } else if (rows >= LARGEMATRIX && cols < LARGEMATRIX) {
            std::vector<size_t> rows_arr(SMALL * 2);
            rows_arr = {0, 1, 2, rows-3, rows-2, rows-1};

            for (const size_t r : rows_arr) {
                for (size_t c = 0; c < cols; ++c) {
                    repr_str += std::to_string(std::round(get_item(r, c) * DECIMALPLACES) / DECIMALPLACES);
                    if (c < cols - 1) {
                        repr_str += ", ";
                    }
                }
                repr_str += "\n";
                if (r == SMALL - 1) {
                    repr_str += "...\n";
                }
            }

        } else if (rows < LARGEMATRIX && cols >= LARGEMATRIX) {
            std::vector<size_t> cols_arr(SMALL * 2);
            cols_arr = {0, 1, 2, cols-3, cols-2, cols-1};
            for (size_t r = 0; r < rows; ++r) {
                for (const size_t c : cols_arr) {
                    repr_str += std::to_string(std::round(get_item(r, c) * DECIMALPLACES) / DECIMALPLACES);
                    if (c < cols - 1) {
                        repr_str += ", ";
                    }
                    if (c == SMALL - 1) {
                        repr_str += "...";
                    }
                }
                repr_str += "\n";
            }
        } else {
            std::vector<size_t> cols_arr(SMALL * 2);
            std::vector<size_t> rows_arr(SMALL * 2);
            rows_arr = {0, 1, 2, rows-3, rows-2, rows-1};
            cols_arr = {0, 1, 2, cols-3, cols-2, cols-1};

            for (const size_t r : rows_arr) {
                for (const size_t c : cols_arr) {
                    repr_str += std::to_string(std::round(get_item(r, c) * DECIMALPLACES) / DECIMALPLACES);
                    if (c < cols - 1) {
                        repr_str += ", ";
                    }
                    if (c == SMALL - 1) {
                        repr_str += "...";
                    }
                }
                repr_str += "\n";
                if (r == SMALL - 1) {
                    repr_str += "...\n";
                }
            }

        }
        */

        return repr_str;
    }

    Matrix& transpose() {
        size_t temp = this->rows;
        this->rows = this->cols;
        this->cols = temp;
        this->data_is_transposed = !this->is_transposed();
        return *this;
    }

    // riyal operations

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
        .def("copy", &Matrix::copy)
        .def("__repr__", &Matrix::repr)
        .def("T", &Matrix::transpose)
        .def("get_array", &Matrix::get_array)
        .def("__getitem__", &Matrix::get_item);
}