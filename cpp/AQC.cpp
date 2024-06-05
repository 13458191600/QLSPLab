#include "QPanda.h"
#include <Eigen/Dense>
#include <iostream>
#include <complex>
#include "block_encoding.cpp"
USING_QPANDA

using namespace Eigen;
using namespace std;

tuple<int, int, int, VectorXd, int> setting(MatrixXd A, VectorXd b, double T, int M) {
    int dimA = A.rows();
    int n = 2 * static_cast<int>(log2(dimA));
    int N = 2 * static_cast<int>(pow(2, n));
    int an = 2 * n + 1;
    b = b / b.norm();
    int d = dimA;
    return make_tuple(N, an, n, b, d);
}

MatrixXcd get_H0(VectorXd b) {
    MatrixXcd Q_b = MatrixXcd::Identity(b.size(), b.size()) - b * b.adjoint();
    MatrixXcd H0 = kroneckerProduct(Matrix2d::Identity() - Matrix2d::Identity() * 0.5, Q_b);
    return H0;
}

MatrixXcd get_H1(MatrixXd A, VectorXd b) {
    MatrixXcd Q_b = MatrixXd::Identity(b.size(), b.size()) - b * b.adjoint();
    MatrixXcd sigmaplus(A.rows(), A.cols());
    sigmaplus << complex<double>(0, 0),complex<double>(1, 0),complex<double>(0, 0),complex<double>(0, 0);
    MatrixXcd sigmaminus(A.rows(), A.cols());
    sigmaminus << complex<double>(0, 0),complex<double>(0, 0),complex<double>(1, 0),complex<double>(0, 0);
    MatrixXcd H1 = kroneckerProduct(sigmaplus, A * Q_b)
                + kroneckerProduct(sigmaminus, Q_b * A);
    return H1;
}
void initialize(QCircuit& qc, QVec& qvec, std::vector<double>& bdata) {
    Encode encode_b;
    encode_b.amplitude_encode(qvec, bdata);
    qc << encode_b.get_circuit();
}
 

void discrete_step(QCircuit & qc, QVec& qvec, MatrixXcd H0, MatrixXcd H1, double f, int n, int M, double T) {
    int dim = H0.rows();
    // Create an identity matrix of complex numbers
    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    MatrixXcd A1 =  I + complex<double>(0, -(1 - f) * f / M * T)*H0;
    MatrixXcd A2 = I - complex<double>(0, -f * f / M * T)*H1;
    // cout << "A1:\n" << A1 << endl;
    // cout << "A2:\n" << A2 << endl;
    // cout << "U1:\n" << BU1 << endl;
    QMatrixXcd U1 = block_encoding_method(A1);
    QMatrixXcd U2 = block_encoding_method(A2);

    qc << matrix_decompose_qr(qvec, U1) << matrix_decompose_qr(qvec, U2); // Example, needs appropriate method
}
void discrete_step_with_trotter(QCircuit& qc, QVec& qvec, MatrixXcd H0, MatrixXcd H1, double f, int n, int M, double T,int DM) {
    int dim = H0.rows();
    // Create an identity matrix of complex numbers
    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    MatrixXcd A1 =  I + complex<double>(0, -(1 - f) * f / (double)M * T/(double)DM)*H0;
    MatrixXcd A2 = I - complex<double>(0, -f * f / (double)M * T/DM)*H0;
    QMatrixXcd U1 = block_encoding_method(A1);
    QMatrixXcd U2 = block_encoding_method(A2);
    qc << matrix_decompose_qr(qvec, U1) << matrix_decompose_qr(qvec, U2);
}

int main() {
    MatrixXd A(2, 2);
    A << 2, 1, 1, 0;
    VectorXd x(2);
    x << 1, 1;
    VectorXd b = A * x;
    std::vector<double>bdata{3,1};
    int T = 800;
    int M = 3;
    auto qvm = CPUQVM();
    qvm.init();
    auto qvec = qvm.qAllocMany(5);
    auto cbits = qvm.cAllocMany(5);
    auto [N, an, n, normalized_b, d] = setting(A, b, T, M);
    MatrixXcd H0 = get_H0(normalized_b);
    MatrixXcd H1 = get_H1(A, normalized_b);
    int b_qubits_num = log2(bdata.size());
    if (bdata.size()!= pow(2,b_qubits_num ))
        b_qubits_num = b_qubits_num+1;
    vector<double> init_state;
    double Lambda = accumulate(bdata.begin(), bdata.end(), 0.0);
    for (int i = 0; i < (int)bdata.size(); ++i) {
        init_state.push_back(sqrt(1.0/Lambda*(bdata[i])));
    }
    for (int i = (int)bdata.size(); i < pow(2,b_qubits_num); ++i) {
        init_state.push_back(0.0);
    }
    QCircuit circuit;
    initialize(circuit, qvec, init_state);  // Placeholder for state initialization
    std::string text_picture = draw_qprog(circuit);
    std::cout << text_picture << std::endl;
    Vector2cd current_state;
    int DM = 200;
    for (double m = 0; m < M; m++) {
        double s = (m + 1) / (double)M;
        for (int i = 0; i < DM; i++) {
            discrete_step_with_trotter(circuit, qvec, H0,H1, s, n, M, T,DM);
            QProg prog;
            prog << circuit;
            QStat cir_matrix = getCircuitMatrix(prog);
            std::string text_picture = draw_qprog(circuit);
            std::cout << text_picture << std::endl;
            cout << "Circuit matrix: " << cir_matrix << endl;
            for (int i = 0; i < 2; i++) {
                current_state[i]= cir_matrix[i*sqrt(cir_matrix.size())];
            }
            current_state = current_state / current_state.norm();
            for (int i = 0; i < 2; i++) {
                init_state[i] = current_state[i].real();
            }
            cout << "Current state: " << current_state << endl;
            QCircuit circuit;
            initialize(circuit, qvec, init_state);
        }
    }
    cout << "Final state: " << current_state << endl;
    return 0;
}
