#include "QPanda.h"
#include <Eigen/Dense>
#include <iostream>
#include <complex>
USING_QPANDA

using namespace Eigen;
using namespace std;

tuple<int, int, int, VectorXd, int, double, int> setting(MatrixXd A, VectorXd b, double T, int M) {
    int dimA = A.rows();
    int n = 2 * static_cast<int>(log2(dimA));
    int N = 2 * static_cast<int>(pow(2, n));
    int an = 2 * n + 1;
    b = b / b.norm();
    int d = dimA;
    return make_tuple(N, an, n, b, d, T, M);
}

MatrixXd get_H0(VectorXd b) {
    MatrixXd Q_b = MatrixXd::Identity(b.size(), b.size()) - b * b.adjoint();
    MatrixXd H0 = kroneckerProduct(Matrix2d::Identity() - Matrix2d::Identity() * 0.5, Q_b).eval();
    return H0;
}
MatrixXcd block_encoding_method(MatrixXcd SparseHamiltonian){
    return SparseHamiltonian;
}

MatrixXd get_H1(MatrixXd A, VectorXd b) {
    MatrixXd Q_b = MatrixXd::Identity(b.size(), b.size()) - b * b.adjoint();
    MatrixXd sigmaplus(A.rows(), A.cols());
    sigmaplus << 0, 1, 0, 0;
    MatrixXd sigmaminus(A.rows(), A.cols());
    sigmaminus << 0, 0, 1, 0;
    MatrixXd H1 = kroneckerProduct(sigmaplus, A * Q_b).eval()
                + kroneckerProduct(sigmaminus, Q_b * A).eval();
    return H1;
}
void initialize(QCircuit& qc, QVec& qvec, vector& bdata) {
    Encode encode_b;
    encode_b.amplitude_encode(qvec, bdata);
    prog << encode_b.get_circuit();
    qc << prog;
}


void discrete_step(QCircuit& qc, QVec& qvec, MatrixXd H0, MatrixXd H1, double f, int n, int M, double T) {
    int dim = H0.rows();
    // Create an identity matrix of complex numbers
    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    MatrixXcd A1 =  I + complex<double>(0, -(1 - f) * f / M * T)*H0;
    MatrixXcd A2 = I - complex<double>(0, -f * f / M * T)*H0;
    MatrixXcd U1 = block_encoding_method(A1);
    MatrixXcd U2 = block_encoding_method(A2);
    qc << matrix_decompose_qr(qvec, U1) << matrix_decompose_qr(qvec, U2); // Example, needs appropriate method
}
void discrete_step_with_trotter(QCircuit& qc, QVec& qvec, MatrixXd H0, MatrixXd H1, double f, int n, int M, double T) {
    int dim = H0.rows();
    // Create an identity matrix of complex numbers
    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    int DM = 200;
    for (int m = 0; m < DM; ++m) {
        double s = static_cast<double>(m + 1) / M;
        MatrixXcd A1 =  I + complex<double>(0, -(1 - f) * f / M * T/DM)*H0;
        MatrixXcd A2 = I - complex<double>(0, -f * f / M * T/DM)*H0;
        MatrixXcd U1 = block_encoding_method(A1);
        MatrixXcd U2 = block_encoding_method(A2);
        qc << matrix_decompose_qr(qvec, U1) << matrix_decompose_qr(qvec, U2);
    }
     // Example, needs appropriate method
}

int main() {
    MatrixXd A(2, 2);
    A << 2, 1, 1, 0;
    VectorXd x(2);
    x << 1, 1;
    VectorXd b = A * x;
    std::vector<double>bdata{3,1};
    int T = 1000;
    int M = 200;
    auto qvm = CPUQVM();
    qvm.init();
    auto qvec = qvm.qAllocMany(4);
    auto cbits = qvm.cAllocMany(4);
    auto[N, an, n, normalized_b, d, T_, M_] = setting(A, b, T, M);
    MatrixXd H0 = get_H0(normalized_b);
    MatrixXd H1 = get_H1(A, normalized_b);
    auto circuit = QCircuit();
    initialize(circuit, qvec,bdata );

    for (int m = 0; m < M; ++m) {
        double s = static_cast<double>(m + 1) / M;
        discrete_step(circuit, qvec, H0,H1, s, n, M, T);
    }
    // Example of running the quantum circuit with QPanda
    
    QProg prog;
    prog << circuit << MeasureAll(qvec, cbits);
    
    auto result = qvm.runWithConfiguration(prog, cbits, 1000);
    for (const auto& val : result) {
        cout << val.first << ", " << val.second << endl;
    }

    return 0;
}
