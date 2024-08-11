#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <omp.h>
#include <string>
#include <cmath>
#include <functional>

#define DECIMALPLACES 1000000
#define LARGEMATRIX 53
#define SMALL 3
#define LARGEMATRIXFORSTRASSEN 64
using namespace std;

// https://cs.stackexchange.com/questions/92666/strassen-algorithm-for-unusal-matrices
// parallelisation thanks to https://github.com/spectre900/Parallel-Strassen-Algorithm/blob/master/omp_strassen.cpp
// https://ppc.cs.aalto.fi/ch3/nested/#:~:text=Parallelizing%20nested%20loops,need%20most%20of%20the%20time.

class Matrix {
    private:
        std::unique_ptr<double[]> mat;
        bool data_is_transposed = false;

        bool is_transposed() const {
            return this->data_is_transposed;
        }
    
    public:
        size_t rows, cols;

    Matrix() : rows(0), cols(0), mat(nullptr), data_is_transposed(false) {}
    
    Matrix(const size_t rows, const size_t cols) : rows(rows), cols(cols) {
        if (rows <= 0 || cols <= 0) {
            throw std::out_of_range("Matrix dimensions must be positive");
        }
        mat = std::make_unique<double[]>(rows * cols);
    }

    Matrix(const size_t rows, const size_t cols, std::unique_ptr<double[]> mat) : rows(rows), cols(cols), mat(std::move(mat)) {
        if (rows <= 0 || cols <= 0) {
            throw std::out_of_range("Matrix dimensions must be positive");
        }
    }

    // copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        mat = std::make_unique<double[]>(rows * cols);

        //#pragma omp parallel for
        for (long i = 0; i < rows * cols; ++i) mat[i] = other.mat[i];

        this->data_is_transposed = other.is_transposed();
    }

    template <typename T>
    Matrix(const T& list);

    static Matrix identity(size_t mat_size) {
        size_t row = mat_size, col = mat_size;
        size_t entries = row * col;
        unique_ptr<double[]> new_mat = std::make_unique<double[]>(entries);

        #pragma omp parallel for
        for (long i = 0; i < row; ++i) {
            for (long j = 0; j < col; ++j) {
                if (i == j) {
                    new_mat[i * col + j] = 1;
                } else {
                    new_mat[i * col + j] = 0;
                }
                
            }
        }
        return Matrix(row, col, std::move(new_mat));
    }

    static Matrix zeroes(size_t row, size_t col) {
        size_t entries = row * col;
        unique_ptr<double[]> new_mat = std::make_unique<double[]>(entries);

        #pragma omp parallel for
        for (long i = 0; i < row; ++i) {
            for (size_t j = 0; j < col; ++j) {
                new_mat[i * col + j] = 0;
            }
        }
        return Matrix(row, col, std::move(new_mat));
    }


    std::vector<double> get_array() const {
        size_t size = rows * cols;
        return std::vector<double>(this->mat.get(), this->mat.get() + size);
    }

    // probably not needed for insanely large matrices
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
    
    std::tuple<size_t, size_t> get_dims() const {
        return std::make_tuple(this->rows, this->cols);
    }

    double get_item(std::tuple<const size_t, const size_t> tup) const {
        size_t r = std::get<0>(tup);
        size_t c = std::get<1>(tup);
        
        return this->get_item_inner(r, c);
    }

    double get_item_inner(size_t r, size_t c) const {
            
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }

        if (this->is_transposed()) {
            return mat[c * rows + r];
        } else {
            return mat[r * cols + c];
        }
        
    }

    void set_item(std::tuple<const size_t, const size_t> tup, double value) {
        size_t r = std::get<0>(tup);
        size_t c = std::get<1>(tup);

        this->set_item_inner(r, c, value);
    }

    void set_item_inner(size_t r, size_t c, double value) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }

        if (this->is_transposed()) {
            this->mat[c * rows + r] = value;
        } else {
            this->mat[r * cols + c] = value;
        }
    }


    Matrix copy() {
        return Matrix(*this);
    }


    string repr() const {
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
                repr_str += std::to_string(std::round(get_item_inner(r, c) * DECIMALPLACES) / DECIMALPLACES);
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


        return repr_str;
    }

    Matrix& transpose() {
        // Also modifies original. (saves time)
        size_t temp = this->rows;
        this->rows = this->cols;
        this->cols = temp;
        this->data_is_transposed = !this->is_transposed();
        return *this;
    }

    // riyal operations
    static Matrix apply_all_entries_mat(const Matrix& curr, const Matrix& other, std::function<double(double, double)> op) {
        if (curr.rows == other.rows && curr.cols == other.cols) {
            size_t entries = curr.rows * curr.cols;
            unique_ptr<double[]> new_mat = std::make_unique<double[]>(entries);
            
            for (long i = 0; i < curr.rows; ++i) {
                for (long j = 0; j < curr.cols; ++j) {
                    new_mat[i * curr.cols + j] = op(curr.get_item_inner(i, j), other.get_item_inner(i, j));
                }
            }
            return Matrix(curr.rows, curr.cols, std::move(new_mat));
        } else {
            throw std::runtime_error("Matrix must have the same dimensions");
        }
    }

    static Matrix apply_all_entries_num(const Matrix& curr, const double num, std::function<double(double, double)> op) {
        size_t entries = curr.rows * curr.cols;
        unique_ptr<double[]> new_mat = std::make_unique<double[]>(entries);

        for (long i = 0; i < curr.rows; ++i) {
            for (long j = 0; j < curr.cols; ++j) {
                new_mat[i * curr.cols + j] = op(curr.get_item_inner(i, j), num);
            }
        }
        return Matrix(curr.rows, curr.cols, std::move(new_mat));
    }

    // ADDING MATRICES

    static Matrix add(const Matrix& matrix, const Matrix& other) {
        return Matrix::apply_all_entries_mat(matrix, other, [](double a, double b) {
            return a + b;
            });
    }

    Matrix add(const Matrix& other) {
        // it's a copy because I want to reduce abstraction overhead blah blah
        return Matrix::apply_all_entries_mat(*this, other, [](double a, double b) {
            return a + b;
            });
    }

    // ADDING NUMBERS

    static Matrix add(const Matrix& matrix, const double number) {
        return Matrix::apply_all_entries_num(matrix, number, [](double a, double b) {
            return a + b;
            });
    }

    Matrix add(const double number) {
        return Matrix::apply_all_entries_num(*this, number, [](double a, double b ) {
            return a + b;
        });
        
    }

    // SUBBING MATRICES

    static Matrix sub(const Matrix& matrix, const Matrix& other) {
        return Matrix::apply_all_entries_mat(matrix, other, [](double a, double b) {
            return a - b;
            });
    }

    Matrix sub(const Matrix& other) {
        return Matrix::apply_all_entries_mat(*this, other, [](double a, double b) {
            return a - b;
            });
    }

    // SUBBING NUMBERS

    static Matrix sub(const Matrix& matrix, const double number) {
        return Matrix::apply_all_entries_num(matrix, number, [](double a, double b) {
            return a - b;
            });
    }


    Matrix sub(const double number) {
        return Matrix::apply_all_entries_num(*this, number, [](double a, double b ) {
            return a - b;
        });
    }

    //hadamard prod

    static Matrix mul(const Matrix& matrix, const Matrix& other) {
        return Matrix::apply_all_entries_mat(matrix, other, [](double a, double b) {
            return a * b;
            });
    }

    Matrix mul(const Matrix& other) {
        return Matrix::apply_all_entries_mat(*this, other, [](double a, double b) {
            return a * b;
            });
    }

    // MUL NUMS

    static Matrix mul(const Matrix& matrix, const double number) {
        return Matrix::apply_all_entries_num(matrix, number, [](double a, double b) {
            return a * b;
            });
    }

    Matrix mul(const double number) {
        return Matrix::apply_all_entries_num(*this, number, [](double a, double b ) {
            return a * b;
        });
    }

    //Wrapper around product
    Matrix neg() {
        return mul(-1);
    }

    bool eq(const Matrix& other) {
        if (this->rows == other.rows && this->cols == other.cols) {
            size_t entries = rows * cols;
            //unique_ptr<double[]> new_mat = make_unique<double[]>(entries);

            for (long i = 0; i < rows; ++i) {
                for (long j = 0; j < cols; ++j) {
                    if (this->get_item_inner(i, j) != other.get_item_inner(i, j)) {
                        return false;
                    }
                }
            }
            return true;
        } else {
            return false;
        }
    }

    Matrix mat_mul_default(const Matrix& other) const {
        const size_t new_rows = this->rows;
        const size_t new_cols = other.cols;
        const size_t entries = new_rows * new_cols;
        unique_ptr<double[]> new_mat = std::make_unique<double[]>(entries);


        for (long i = 0; i < new_rows; ++i) {
            for (long j = 0; j < new_cols; ++j) {
                double temp = 0;
                for (size_t k = 0; k < this->cols; ++k) {
                    temp += this->get_item_inner(i, k) * other.get_item_inner(k, j);
                }
                new_mat[i * new_cols + j] = temp;
            }
        }
        return Matrix(new_rows, new_cols, std::move(new_mat));

    }

    // pads to the smallest power of 2 greater than length
    Matrix pad_matrix_to_2n(size_t length) const {
        // length is greater than this->cols and this->rows
        size_t result = 1;
        while (result < length) {
            result <<= 1;
        }

        Matrix padded = Matrix::zeroes(result, result);

        for (long e = 0; e < this->rows * this->cols; ++e) {
            long r = e / this->cols;
            long c = e % this->cols;
            padded.set_item_inner(r, c, this->get_item_inner(r, c));
        }


        // long r, c;
        // #pragma omp parallel for private(r, c)
        // for (r = 0; r < this->rows; ++r) {
        //     for (c = 0; c < this->cols; ++c) {
        //         padded.set_item_inner(r, c, this->get_item_inner(r, c));
        //     }
        // }
        return padded;
    }

    // static method, must be padded to square matrix
    static std::tuple<Matrix, Matrix, Matrix, Matrix> get_quadrants(const Matrix& padded) {
        size_t length = padded.rows >> 1; // 2 * length == this->cols;
        unique_ptr<double[]> mat_1 = std::make_unique<double[]>(length * length);
        unique_ptr<double[]> mat_2 = std::make_unique<double[]>(length * length);
        unique_ptr<double[]> mat_3 = std::make_unique<double[]>(length * length);
        unique_ptr<double[]> mat_4 = std::make_unique<double[]>(length * length);
        
        for (long i = 0; i < length; ++i) {
            for (long j = 0; j < length; ++j) {
                mat_1[i * length + j] = padded.get_item_inner(i, j);
                mat_2[i * length + j] = padded.get_item_inner(i, j + length);
                mat_3[i * length + j] = padded.get_item_inner(i + length, j);
                mat_4[i * length + j] = padded.get_item_inner(i + length, j + length);
            }
        }

        return std::make_tuple(
            Matrix(length, length, std::move(mat_1)),
            Matrix(length, length, std::move(mat_2)),
            Matrix(length, length, std::move(mat_3)),
            Matrix(length, length, std::move(mat_4))
        );
    }

    static Matrix combine(const Matrix& C11, const Matrix& C12, const Matrix& C21, const Matrix& C22) {
        // number of rows must all be the same
        const size_t length = C11.rows; // desired length is twice of that
        const size_t desired = 2 * length;
        unique_ptr<double[]> combined = std::make_unique<double[]>(desired * desired);

        for (long r = 0; r < desired; ++r) {
            for (long c = 0; c < desired; ++c) {
                if (r < length && c < length) {
                    combined[r * desired + c] = C11.get_item_inner(r, c);
                } else if (r < length) {
                    combined[r * desired + c] = C12.get_item_inner(r, c - length);
                } else if (c < length) {
                    combined[r * desired + c] = C21.get_item_inner(r - length, c);
                } else {
                    combined[r * desired + c] = C22.get_item_inner(r - length, c - length);
                }
                
            }
        }
        return Matrix(desired, desired, std::move(combined));
    }
    // static method, must be padded
    static Matrix strassen(const Matrix& a_padded, const Matrix& b_padded) {
        // https://gist.github.com/syphh/1cb6b9bb57a400873fa9d05cd1ee7cc3
        if (a_padded.rows < LARGEMATRIXFORSTRASSEN) {
            return a_padded.mat_mul_default(b_padded);
        }
        const auto& tuple_1 = Matrix::get_quadrants(a_padded);
        const auto& tuple_2 = Matrix::get_quadrants(b_padded);

        const Matrix& A = std::get<0>(tuple_1);
        const Matrix& B = std::get<1>(tuple_1);
        const Matrix& C = std::get<2>(tuple_1);
        const Matrix& D = std::get<3>(tuple_1);
        // const auto& [A, B, C, D] = tuple_1; // but im not using C++ 17
        
        const Matrix& E = std::get<0>(tuple_2);
        const Matrix& F = std::get<1>(tuple_2);
        const Matrix& G = std::get<2>(tuple_2);
        const Matrix& H = std::get<3>(tuple_2);

        Matrix P1, P2, P3, P4, P5, P6, P7;

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                P1 = Matrix::strassen(Matrix::add(A, D), Matrix::add(E, H));

                #pragma omp task
                P2 = Matrix::strassen(D, Matrix::sub(G, E));

                #pragma omp task 
                P3 = Matrix::strassen(Matrix::add(A, B), H);

                #pragma omp task
                P4 = Matrix::strassen(Matrix::sub(B, D), Matrix::add(G, H));

                #pragma omp task 
                P5 = Matrix::strassen(A, Matrix::sub(F, H));

                #pragma omp task
                P6 = Matrix::strassen(Matrix::add(C, D), E);

                #pragma omp task
                P7 = Matrix::strassen(Matrix::sub(A, C), Matrix::add(E, F));
                #pragma omp taskwait
            }
            
        }

        const Matrix& C11 = Matrix::add(Matrix::add(P1, P2), Matrix::sub(P4, P3));
        const Matrix& C12 = Matrix::add(P5, P3);
        const Matrix& C21 = Matrix::add(P2, P6);
        const Matrix& C22 = Matrix::sub(Matrix::add(P1, P5), Matrix::add(P6, P7));


        // combine
        return Matrix::combine(C11, C12, C21, C22);
    }

    // uses strassen's algorithm
    Matrix mat_mul(const Matrix& other) const {
        if (this->cols != other.rows) {
            throw std::runtime_error(
                "Dimensions of " + std::to_string(this->cols) + " and " + std::to_string(other.rows) +  " do not match"
            );
        }

        if (this->cols < LARGEMATRIXFORSTRASSEN || this->rows < LARGEMATRIXFORSTRASSEN || other.cols < LARGEMATRIXFORSTRASSEN) {
            return mat_mul_default(other);
        } else {
            // pad both then mult
            size_t length = std::max(std::max(this->cols, this->rows), other.cols);
            const Matrix this_padded = this->pad_matrix_to_2n(length);
            const Matrix other_padded = other.pad_matrix_to_2n(length);
            Matrix padded_result = strassen(this_padded, other_padded);
            //remove padding
            unique_ptr<double[]> unpadded = std::make_unique<double[]>(this->rows * other.cols);
            
            
            #pragma omp parallel for
            for (long e = 0; e < this->rows * other.cols; ++e) {
                long i = e / other.cols;
                long j = e % other.cols;
                unpadded[i * other.cols + j] = padded_result.get_item_inner(i, j);
            }


            // long i, j;
            // #pragma omp parallel for private(i, j)
            // for (i = 0; i < this->rows; ++i) {
            //     for (j = 0; j < other.cols; ++j) {
            //         unpadded[i * other.cols + j] = padded_result.get_item_inner(i, j);
            //     }
            // }

            return Matrix(this->rows, other.cols, std::move(unpadded));
        }
        
    }

    Matrix pow(long number) {
        if (this->cols != this->rows) {
            throw std::runtime_error("Matrix must be square");
        }

        if (number == 0) {
            return Matrix::identity(this->rows);
        } else if (number == 1) {
            return *this;
        } else if (number > 1) {
            std::vector<bool> vec;
            while (number > 1) {
                vec.push_back(bool(number & 0x1));
                number >>= 1;
            }
            vec.push_back(1);
            Matrix curr = Matrix::identity(this->rows);
            Matrix multiplier = *this;
            for (const bool element : vec) {
                if (element) {
                    curr = curr.mat_mul(multiplier);
                }
                multiplier = multiplier.mat_mul(multiplier);
            }
            return curr; 
        } else {
            throw std::logic_error("Inverse not yet implemented");
        }
    }

};