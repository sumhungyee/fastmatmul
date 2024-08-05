#include "matmul.h"

/* References for self

https://stackoverflow.com/questions/49301317/pybind11-passing-a-python-list-to-c-style-array
https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
https://stackoverflow.com/questions/60745723/pybind11-wrapping-overloaded-assignment-operator

*/



int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(matmul, m) {
    m.doc() = "A fun module I built while learning cpp, wip"; // still in the works
    m.def("add", &add, "A function that adds two numbers");

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<const py::list&>())
        .def("assign", &Matrix::operator=)
        .def("copy", &Matrix::copy)
        .def("__repr__", &Matrix::repr)
        .def("T", &Matrix::transpose)
        .def("get_array", &Matrix::get_array)
        .def("__getitem__", &Matrix::get_item)
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
        ;
}