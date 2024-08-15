#include <algorithm>
#include <assert.h>
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

#define STRASSEN_POWER 6
#define LARGEMATRIXFORSTRASSEN 1 << STRASSEN_POWER
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

        static size_t get_2n(size_t length) {
            size_t count = 0;
            while (length > LARGEMATRIXFORSTRASSEN) {
                if ((length & 1) == 1) {
                    length += 1;
                }
                length >>= 1;
                count += 1; 
            }
            return length << count;
        }

        // used to skip transpose checks i.e. matrix just created.
        void set_item_inner_assume_no_t(size_t r, size_t c, double value) {
            this->mat[r * cols + c] = value;
        }

        double get_item_inner(size_t r, size_t c) const {

            if (this->data_is_transposed) { // reduce overhead
                return mat[c * rows + r];
            } else {
                return mat[r * cols + c];
            }
        }

        void set_item_inner(size_t r, size_t c, double value) {

            if (this->data_is_transposed) { // reduce overhead
                this->mat[c * rows + r] = value;
            } else {
                this->mat[r * cols + c] = value;
            }
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

        #pragma omp parallel for shared(new_mat)
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
        for (long i = 0; i < entries; ++i) {
            new_mat[i] = 0;
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
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }
        
        return this->get_item_inner(r, c);
    }


    void set_item(std::tuple<const size_t, const size_t> tup, double value) {
        size_t r = std::get<0>(tup);
        size_t c = std::get<1>(tup);
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Matrix index out of bounds");
        }

        this->set_item_inner(r, c, value);
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

    static void fastadd(const Matrix& curr, const Matrix& other, Matrix& result) {
        // removes checks and assume untranspose, to be used only in strassens
        const size_t entries = curr.rows * curr.cols;
        for (size_t i = 0; i < entries; ++i) {
            // matrices guaranteed to not be transposed due to padding
            result.mat[i] = curr.mat[i] + other.mat[i];
        }
    }

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

    static void fastsub(const Matrix& curr, const Matrix& other, Matrix& result) {
        const size_t entries = curr.rows * curr.cols;
        for (size_t i = 0; i < entries; ++i) {
            result.mat[i] = curr.mat[i] - other.mat[i];
        }
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


    // instead of padding to the smallest power of 2 greater than length
    // pad to 2^n * m where m <= LARGEMATRIXFORSTRASSEN
    Matrix pad_matrix_to_2n(size_t length) const {
        // length is greater than this->cols and this->rows
        size_t result = Matrix::get_2n(length);
        Matrix padded = Matrix::zeroes(result, result);

        #pragma omp parallel for
        for (long e = 0; e < this->rows * this->cols; ++e) {
            long r = e / this->cols;
            long c = e % this->cols;
            // padded is untransposed
            padded.set_item_inner_assume_no_t(r, c, this->get_item_inner(r, c));
        }
        // padded is guaranteed to be untransposed
        assert(!padded.is_transposed());
        return padded;
    }

    // static method, must be padded to square matrix
    static std::tuple<Matrix, Matrix, Matrix, Matrix> get_quadrants(const Matrix& padded) {
        // padded is guaranteed to be untransposed
        size_t length = padded.rows >> 1; // 2 * length == this->cols;
        size_t dims = length * length;
        unique_ptr<double[]> mat_1 = std::make_unique<double[]>(dims);
        unique_ptr<double[]> mat_2 = std::make_unique<double[]>(dims);
        unique_ptr<double[]> mat_3 = std::make_unique<double[]>(dims);
        unique_ptr<double[]> mat_4 = std::make_unique<double[]>(dims);

        size_t location, value1, value2;
        for (long i = 0; i < length; ++i) {
            for (long j = 0; j < length; ++j) {
                // reduce overhead with needless checks
                location = i * length + j;
                value1 = i * padded.cols + j;
                value2 = (i + length) * padded.cols + j;
                // due to padding from strassens, matrix is guaranteed to not be transposed
                mat_1[location] = padded.mat[value1];
                mat_2[location] = padded.mat[value1 + length];
                mat_3[location] = padded.mat[value2];
                mat_4[location] = padded.mat[value2 + length];
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

        // again, guaranteed untransposed
        // return mat[r * cols + c];
        for (long r = 0; r < desired; ++r) {
            for (long c = 0; c < desired; ++c) {
                if (r < length && c < length) {
                    combined[r * desired + c] = C11.mat[r * length + c];
                } else if (r < length) {
                    combined[r * desired + c] = C12.mat[r * length + c - length];
                } else if (c < length) {
                    combined[r * desired + c] = C21.mat[(r - length) * length + c];
                } else {
                    combined[r * desired + c] = C22.mat[(r - length) * length + c - length];
                }
            }
        }
        return Matrix(desired, desired, std::move(combined));
    }
    // static method, must be padded
    static Matrix strassen(const Matrix& a_padded, const Matrix& b_padded) {
        // this function was changed to reduce copies made. instead, fast_add and
        // fast_sub now take in a scratch matrix which can be edited.
        if (a_padded.rows < LARGEMATRIXFORSTRASSEN) {
            return a_padded.mat_mul_default(b_padded);
        }
        const auto& tuple_1 = Matrix::get_quadrants(a_padded);
        const auto& tuple_2 = Matrix::get_quadrants(b_padded);

        const Matrix& A = std::get<0>(tuple_1);
        const Matrix& B = std::get<1>(tuple_1);
        const Matrix& C = std::get<2>(tuple_1);
        const Matrix& D = std::get<3>(tuple_1);

        const Matrix& E = std::get<0>(tuple_2);
        const Matrix& F = std::get<1>(tuple_2);
        const Matrix& G = std::get<2>(tuple_2);
        const Matrix& H = std::get<3>(tuple_2);

        Matrix P1, P2, P3, P4, P5, P6, P7;

        size_t halved_dim = a_padded.rows >> 1;

        Matrix result1(halved_dim, halved_dim);
        Matrix result2(halved_dim, halved_dim);

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task shared(P1)
                {
                    Matrix::fastadd(A, D, result1);
                    Matrix::fastadd(E, H, result2);
                    P1 = Matrix::strassen(result1, result2);
                }
                

                #pragma omp task shared(P2)
                {
                    Matrix result3(halved_dim, halved_dim);
                    Matrix::fastsub(G, E, result3);
                    P2 = Matrix::strassen(D, result3);
                }
                

                #pragma omp task shared(P3) 
                {
                    Matrix result4(halved_dim, halved_dim);
                    Matrix::fastadd(A, B, result4);
                    P3 = Matrix::strassen(result4, H);
                }
                

                #pragma omp task shared(P4)
                {
                    Matrix result5(halved_dim, halved_dim);
                    Matrix result6(halved_dim, halved_dim);
                    Matrix::fastsub(B, D, result5);
                    Matrix::fastadd(G, H, result6);
                    P4 = Matrix::strassen(result5, result6);
                }
                

                #pragma omp task shared(P5)
                {
                    Matrix result7(halved_dim, halved_dim);
                    Matrix::fastsub(F, H, result7);
                    P5 = Matrix::strassen(A, result7);
                }

                #pragma omp task shared(P6)
                {
                    Matrix result8(halved_dim, halved_dim);
                    Matrix::fastadd(C, D, result8);
                    P6 = Matrix::strassen(result8, E);
                }
                

                #pragma omp task shared(P7)
                {
                    Matrix result9(halved_dim, halved_dim);
                    Matrix result10(halved_dim, halved_dim);
                    Matrix::fastsub(A, C, result9);
                    Matrix::fastadd(E, F, result10);
                    P7 = Matrix::strassen(result9, result10);
                }
                
                #pragma omp taskwait
            }
        }

        Matrix C11(halved_dim, halved_dim);
        Matrix C12(halved_dim, halved_dim);
        Matrix C21(halved_dim, halved_dim);
        Matrix C22(halved_dim, halved_dim);

        Matrix::fastadd(P1, P2, result1);
        Matrix::fastsub(P4, P3, result2);
        Matrix::fastadd(result1, result2, C11);

        Matrix::fastadd(P5, P3, C12);

        Matrix::fastadd(P2, P6, C21);

        Matrix::fastadd(P1, P5, result1);
        Matrix::fastadd(P6, P7, result2);
        Matrix::fastsub(result1, result2, C22);

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
                unpadded[i * other.cols + j] = padded_result.mat[i * padded_result.cols + j]; //guaranteed untransposed
            }

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