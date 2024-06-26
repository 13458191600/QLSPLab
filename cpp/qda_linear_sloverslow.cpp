#include "QPanda.h"
#include <Eigen/Dense>
#include <iostream>
#include <complex>
#include "block_encoding.cpp"
#include "initialize.h"
#include <fstream>
USING_QPANDA

using namespace Eigen;
using namespace std;
typedef complex<double> Complex;


struct qdals_res{
    VectorXcd state; 
    complex<double> fidelity;
    qdals_res(VectorXcd s, complex<double> f): state(s), fidelity(f) {}
};

MatrixXcd get_H0(VectorXcd b) {
    MatrixXcd Q_b = MatrixXcd::Identity(b.size(), b.size()) - b * b.adjoint();
    MatrixXcd sigmax(2, 2);
    sigmax << Complex(0, 0), Complex(1, 0),
        Complex(1, 0), Complex(0, 0);
    MatrixXcd H0 = kroneckerProduct(sigmax, Q_b);
    return H0;
}

MatrixXcd get_H1(MatrixXcd A, VectorXcd b) {
    MatrixXcd Q_b = MatrixXcd::Identity(b.size(), b.size()) - b * b.adjoint();
    MatrixXcd sigmaplus(2, 2);
    sigmaplus << Complex(0, 0), Complex(1, 0),
              Complex(0, 0), Complex(0, 0);
    MatrixXcd sigmaminus(2, 2);
    sigmaminus << Complex(0, 0), Complex(0, 0), 
              Complex(1, 0), Complex(0, 0);
    MatrixXcd H1 = kroneckerProduct(sigmaplus, A * Q_b)
                + kroneckerProduct(sigmaminus, Q_b * A);
    return H1;
}


void discrete_step(QCircuit & qc, QVec& qvec, MatrixXcd H0, MatrixXcd H1, double f, int n, int M, double T) {
    int dim = H0.rows();
    // Create an identity matrix of complex numbers
    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    MatrixXcd A1 = I + Complex(0, -(1 - f) * f / M * T) * H0;
    MatrixXcd A2 = I - Complex(0, -f * f / M * T) * H1;
    QMatrixXcd U1 = block_encoding_method(A1);
    QMatrixXcd U2 = block_encoding_method(A2);
    qc << matrix_decompose_qr(qvec, U1) << matrix_decompose_qr(qvec, U2);
}

void discrete_step_with_trotter(QCircuit& qc, QVec& qvec, MatrixXcd H0, MatrixXcd H1, double f, int M, double T, int DM) {
    int dim = H0.rows();
    // Create an identity matrix of complex numbers
    MatrixXcd I = MatrixXcd::Identity(dim, dim);
    MatrixXcd A1 = I + Complex(0, -(1 - f) * f / (double)M * T/(double)DM) * H0;
    MatrixXcd A2 = I + Complex(0, -f * f / (double)M * T/(double)DM) * H1;
    QMatrixXcd U1 = block_encoding_method(A1);
    QMatrixXcd U2 = block_encoding_method(A2);
    qc << matrix_decompose_qr(qvec, U1) << matrix_decompose_qr(qvec, U2);
}


qdals_res qda_linear_solver(MatrixXcd A, VectorXcd b, double T=100, int M=200) {
    VectorXcd target = A.colPivHouseholderQr().solve(b);
    std::ofstream outFile("outputcpp.log");
    if (!outFile) {
        std::cerr << "无法打开文件!" << std::endl;
        return qdals_res(VectorXcd(), 0);
    }

    // if A shape is not 2^n, pad it with zeros and also pad b with zeros
    int dimA = A.rows(); // the dimension of A
    int n = static_cast<int>(log2(dimA)); // the number of qubits of single A
    int N = static_cast<int>(pow(2, n)); // the dimension of single A
    if (dimA != N) { // pad A and b with zeros
        n = n + 1;
        N = pow(2, n);
        MatrixXcd A_padded = MatrixXcd::Zero(N, N);
        A_padded.block(0, 0, A.rows(), A.cols()) = A;
        A = A_padded;
        VectorXcd b_padded = VectorXcd::Zero(N);
        b_padded.head(b.size()) = b;
        b = b_padded;
    }
    b = b / b.norm();
    int aqc_n = 2 * n;
    int aqc_N = pow(2, aqc_n);
    int all_n = 2 * aqc_n + 1;
    MatrixXcd H0 = get_H0(b);
    MatrixXcd H1 = get_H1(A, b);
    assert(H0.rows() == aqc_N);
    assert(H1.rows() == aqc_N);
    auto qvm = new CPUQVM();
    qvm->init();
    auto qvec = qvm->qAllocMany(all_n);
    auto cbits = qvm->cAllocMany(all_n);
    cout << "The number of qubits is: " << all_n << endl;
    // initialize the state by b
    // get the input initial state
    VectorXcd current_state = VectorXcd::Zero( b.size()* b.size());
    current_state.head(b.size()) = b;
    current_state.normalize();
    int DM = 200;
    for (double m = 0; m < M; m++) {
        double s = (m + 1) / (double)M;
        outFile << "The current s is: " << s << endl;
        for (int i = 0; i < DM; i++) {
            QProg circuit;
            // cout << "Iteration: " << m << " " << i << endl;
            // cout << "The current solution is: " << current_state << endl;
            for (int j = 0; j < current_state.size(); j++) {
                outFile << current_state[j] << " ";
                cout << current_state[j] << " ";
            }
            outFile << endl;
            cout << endl;
            // cout << "The norm of the quantum state is: " << current_state.norm() << endl;
            QVec qvec_tmp = QVec(qvec.end() - aqc_n ,qvec.end());
            initialize(circuit, qvec_tmp, current_state);
            circuit << X(qvec[all_n-1]) << X(qvec[all_n-1]);
            VectorXcd inistate = getQuantumStates(qvm, circuit );
            // cout << "The quantum state is: " << inistate << endl;
            int dim = H0.rows();
            double f = s;
            MatrixXcd I = MatrixXcd::Identity(dim, dim);

            MatrixXcd A1 = I + Complex(0, -f  / (double)M * T/(double)DM) * H1;

            QMatrixXcd U1 = block_encoding_method(A1);
            // cout << "The matrix A1 is: \n" << A1 << endl;
            // cout << "The matrix U1 is: \n" << U1.block(0, 0, 4, 4) << endl;
            circuit << matrix_decompose_qr(qvec, U1); // 有可能是加反了
            VectorXcd state = getQuantumStates(qvm, circuit );
            current_state = state.head(2* b.size());
            current_state.normalize();
            for (int j = 0; j < current_state.size(); j++) {
                outFile << current_state[j] << " ";
                cout << current_state[j] << " ";
            }
            outFile << endl;
            cout << endl;
            // cout << "The norm of the quantum state is: " << current_state.norm() << endl;
            QProg secendcircuit;
            initialize(secendcircuit, qvec_tmp, current_state);
            MatrixXcd A0 = I + Complex(0, -(1 - f) / (double)M * T/(double)DM) * H0;
            QMatrixXcd U0 = block_encoding_method(A0);
            secendcircuit << matrix_decompose_qr(qvec, U0); // 有可能是加反了
            state = getQuantumStates(qvm, secendcircuit );
            current_state = state.head(2* b.size());
            current_state = current_state/ current_state.norm();
            outFile.flush();
            // print the state norm
        }
    }
    target.normalize();
    VectorXcd final_state = current_state.head(dimA);
    final_state.normalize();
    complex<double> dotproduct = final_state.dot(target);
    complex<double> distance = abs(dotproduct);
    return qdals_res(final_state, distance);
}

int main() {
    MatrixXcd A(2, 2);
    A << Complex(2, 0), Complex(1, 0), Complex(1, 0), Complex(0, 0);
    VectorXcd b(2);
    b << Complex(3, 0), Complex(1, 0);
    cout << "The matrix A is: \n" << A << endl;
    cout << "The vector b is: \n" << b << endl;
    qdals_res result = qda_linear_solver(A, b);
    cout << "The state is: " << result.state << endl;
    cout << "The distance is: " << result.fidelity << endl;
}
