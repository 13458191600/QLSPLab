/* #include <iostream>
#include <Eigen/Dense>
// #include <unsupported/Eigen/MatrixFunctions>  // For sqrtm
#include </usr/include/eigen3/unsupported/Eigen/MatrixFunctions>
#include <complex>

using namespace std;
using namespace Eigen;

typedef Matrix<std::complex<double>, Dynamic, Dynamic> ComplexMatrix;

int main() {
    // Define identity matrix I
    ComplexMatrix I = ComplexMatrix::Identity(2, 2);

    // Define test matrix A
    ComplexMatrix A(2, 2);
    A << complex<double>(0.1, 0), complex<double>(0.2, 0),
         complex<double>(0.3, 0), complex<double>(0.4, 0);

    // Uncomment to use other test matrices
    // A << complex<double>(1, 0), complex<double>(2, 0),
    //      complex<double>(3, 0), complex<double>(4, 0);
    // A << complex<double>(0.5, 0), complex<double>(-0.9, 0),
    //      complex<double>(0.8, 0), complex<double>(0.5, 0);

    // Check the 2-norm of matrix A
    if (A.norm() > 1) {
        throw runtime_error("Matrix 2-norm greater than 1");
    }

    // Calculate B, C, and D
    ComplexMatrix B = (I - A * A.adjoint()).sqrt();
    ComplexMatrix C = (I - A.adjoint() * A).sqrt();
    ComplexMatrix D = -A.adjoint();

    // Concatenate matrices AB and CD
    ComplexMatrix AB(2, 4);
    AB << A, B;

    ComplexMatrix CD(2, 4);
    CD << C, D;

    // Combine AB and CD to form U
    ComplexMatrix U(4, 4);
    U << AB,
         CD;

    // Print matrices
    cout << "Matrix A:" << endl << A << endl;
    cout << "Matrix U[:2,:2]:" << endl << U.block(0, 0, 2, 2) << endl;

    // Verify unitarity of U
    ComplexMatrix U_dagger_U = (U.adjoint() * U).eval();
    cout << "Verification of unitarity (U†U):" << endl << round(U_dagger_U) << endl;

    return 0;
} */

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
// #include </usr/include/eigen3/unsupported/Eigen/MatrixFunctions>
#include <math.h>
#include <qaoaeancoding.h>

using namespace Eigen;
using namespace std;


int main(){
    // typedef MatrixXcf<float, 2, 2> MatrixXcf_m;
    // 定义矩阵I
    // MatrixXcf I(2, 2);
    VectorXd I(2, 2);
    I << 1, 0,
         0, 1;

    // 定义矩阵A
    // MatrixXcf A(2, 2);
    VectorXd A(2, 2);
    A << 0.1, 0.2,
         0.3, 0.4;
    cout << "A:\n" << A << endl;

    // // 检查矩阵A的二范数
    double normA = A.norm();
    if (normA > 1) {
        cerr << "矩阵的二范数为" << normA << "，大于1" << endl;
        return 1;
    }

    // 计算矩阵B
    auto At = A.transpose();
    // MatrixXcf B = (I - A * At.conjugate()).array().sqrt();
    VectorXd B = (I - A * At.conjugate()).array().sqrt();
    cout << "B:\n" << B << endl;

    // // 计算矩阵C和D
    // MatrixXcd C = (I - A.transpose().conjugate() * A);
    // .array().sqrt();
    // MatrixXcd D = -At.conjugate();

    // // 合并矩阵AB和CD
    // MatrixXcd AB = MatrixXcd::Zero(2, 4);
    // AB << A, B;
    // MatrixXcd CD = MatrixXcd::Zero(2, 4);
    // CD << C, D;

    // // 合并U矩阵
    // MatrixXcd U = MatrixXcd::Zero(4, 4);
    // U << AB, CD;

    // // 输出结果
    // cout << "矩阵A:\n" << A << endl;
    // cout << "U[:2,:2]:\n" << U.block<2,2>(0,0) << endl;
    // cout << "(U.transpose().conj() @ U).round(10):\n" << round((U.transpose().conjugate() * U)) << endl;
    // cout << "(U.transpose().conj() @ U).round(10):\n" << round((U.transpose().conjugate() * U)) << endl;

    return 0;
} 