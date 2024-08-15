#include "../src/matmul.h" 
int main() {
    Matrix m1 = Matrix::identity(100);
    Matrix m2 = Matrix::identity(100);
    std::cout << Matrix::strassen(m1, m2).repr() << std::endl; 
    //std::cout << "hi" << std::endl; 
}