#include <armadillo>
#include <iostream>

int main() {
    // 定义 3x3 矩阵
    arma::mat A = {{1,2,3},{4,5,6},{7,8,9}};
    // 定义向量
    arma::mat B = {{1,0,0},{0,1,0},{0,0,1}};

    arma::mat C = A * B;
    C.print("A * B = ");

    return 0;
}
