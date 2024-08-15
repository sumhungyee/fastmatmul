#include "matmul.h" 
int main() {
    Matrix m1 = Matrix::identity(100);
    Matrix m2 = Matrix::identity(100);
    Matrix::strassen(m1, m2); 
}