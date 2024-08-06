#include "matmul.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>



/* References for self

https://stackoverflow.com/questions/49301317/pybind11-passing-a-python-list-to-c-style-array
https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
https://stackoverflow.com/questions/60745723/pybind11-wrapping-overloaded-assignment-operator

*/

namespace py = pybind11;
template <typename T>
Matrix::Matrix(const T& list) {
        
    this->rows = list.size();
    if (list.empty()) {
        throw std::runtime_error("Matrix must be nonempty");
    } else if (!py::isinstance<py::list>(list.attr("__getitem__")(0)) && !py::isinstance<py::tuple>(list.attr("__getitem__")(0))) {
        auto temp = py::list();
        temp.append(list);
        T outer = temp.cast<T>();
        *this = Matrix(outer);
        return;
    }

    //std::vector<std::vector<double>> matrix_cast = list.cast<std::vector<std::vector<double>>>();
    const auto& matrix_cast = list.cast<std::vector<std::vector<double>>>();
    
    
    auto first_cast = matrix_cast[0];
    this->cols = first_cast.size();
    this->mat = make_unique<double[]>(rows * cols);
    // copy first row in
    std::copy(first_cast.begin(), first_cast.end(), this->mat.get());
    for (size_t i = 1; i < rows; ++i) {
        auto casted = matrix_cast[i];
        if (casted.size() != this->cols) {
            throw std::runtime_error("Matrix rows have different lengths!");
        }
        std::copy(casted.begin(), casted.end(), this->mat.get() + i * cols);
    }
}

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(matmul, m) {
    m.doc() = "A fun module I built while learning cpp, wip"; // still in the works
    //m.def("add", &add, "A function that adds two numbers");

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<const py::list&>())
        .def(py::init<const py::tuple&>())
        .def("assign", &Matrix::operator=)
        .def("T", &Matrix::transpose)
        .def("copy", &Matrix::copy)
        .def("__repr__", &Matrix::repr)
        .def("__getitem__", &Matrix::get_item)
        .def("__setitem__", &Matrix::set_item)
        .def("__add__", py::overload_cast<const Matrix&>(&Matrix::add))
        .def("__add__", py::overload_cast<const double>(&Matrix::add))
        .def("__radd__", py::overload_cast<const double>(&Matrix::add))
        .def("__sub__", py::overload_cast<const Matrix&>(&Matrix::sub))
        .def("__sub__", py::overload_cast<const double>(&Matrix::sub))
        .def("__rsub__", py::overload_cast<const double>(&Matrix::sub))
        .def("__mul__", py::overload_cast<const Matrix&>(&Matrix::mul))
        .def("__mul__", py::overload_cast<const double>(&Matrix::mul))
        .def("__rmul__", py::overload_cast<const double>(&Matrix::mul))
        .def("__neg__", &Matrix::neg)
        .def("__eq__", &Matrix::eq)
        .def("__get_underlying__", &Matrix::get_array)
       
        ;
}