#include <iostream>
#include <Eigen/Dense>
#include <QPanda.h>
#include <cmath>
#include <vector>
USING_QPANDA

using namespace std;
using namespace Eigen;
// using namespace QPanda;

int gray_code(int b) {
    return b ^ (b >> 1);
}

VectorXcd gray_permutation(const VectorXcd& a) {
    VectorXcd b = VectorXcd::Zero(a.size());
    for (int i = 0; i < a.size(); ++i) {
        b[i] = a[gray_code(i)];
    }
    return b;
}

VectorXcd sfwht(VectorXcd a) {
    int n = static_cast<int>(log2(a.size()));
    for (int h = 0; h < n; ++h) {
        for (int i = 0; i < a.size(); i += (1 << (h + 1))) {
            for (int j = i; j < i + (1 << h); ++j) {
                complex<double> x = a[j];
                complex<double> y = a[j + (1 << h)];
                a[j] = (x + y) / 2.0;
                a[j + (1 << h)] = (x - y) / 2.0;
            }
        }
    }
    return a;
}

int compute_control(int i, int n) {
    if (i == pow(4, n)) {
        return 1;
    }
    return 2 * n - static_cast<int>(log2(gray_code(i - 1) ^ gray_code(i)));
}

QCircuit compressed_uniform_rotation(const VectorXcd& a, bool ry = true) {
    int n = static_cast<int>(log2(a.size()) / 2);
    auto qvm = initQuantumMachine(QMachineType::CPU);
    QCircuit circ;
    int num_q = 2 * n + 1;
    auto qubits = qvm->allocateQubits(2 * n + 1);

    int i = 0;
    while (i < a.size()) {
        int parity_check = 0;

        if (a[i] != complex<double>(0, 0)) {
            if (ry) {
                circ << RY(qubits[num_q - 1], a[i].real());  // 使用实部
            } else {
                circ << RZ(qubits[num_q - 1], a[i].real());  // 使用实部
            }
        }

        while (true) {
            int ctrl = compute_control(i + 1, n);
            parity_check ^= (1 << (ctrl - 1));
            i++;
            if (i >= a.size() || a[i] != complex<double>(0, 0)) {
                break;
            }
        }

        for (int j = 1; j <= 2 * n; ++j) {
            if (parity_check & (1 << (j - 1))) {
                circ << CNOT(qubits[num_q - 1 - j], qubits[num_q - 1]);
            }
        }
    }

    return circ;
}

// FABLE
pair<QCircuit, double> fable(MatrixXcd a, double epsilon = -1) {
    double epsm = numeric_limits<double>::epsilon();
    double alpha = a.cwiseAbs().maxCoeff();
    if (alpha > 1) {
        alpha += sqrt(epsm);
        a /= alpha;
    } else {
        alpha = 1.0;
    }

    int n = a.rows();
    int logn = static_cast<int>(ceil(log2(n)));
    if (n < (1 << logn)) {
        MatrixXcd padded = MatrixXcd::Zero(1 << logn, 1 << logn);
        padded.topLeftCorner(n, n) = a;
        a = padded;
    }

    VectorXcd vec_a = Map<VectorXcd>(a.data(), a.size());
    vec_a = gray_permutation(sfwht(2.0 * vec_a.array().acos().matrix()));

    if (epsilon >= 0) {
        for (int i = 0; i < vec_a.size(); ++i) {
            if (abs(vec_a[i]) <= epsilon) {
                vec_a[i] = complex<double>(0, 0);
            }
        }
    }

    QCircuit OA = compressed_uniform_rotation(vec_a);
    QCircuit circ;
    auto qvm = new CPUQVM();
    qvm->init();
    int num_q = 2 * logn + 1;
    auto qubits = qvm->qAllocMany(2 * logn + 1);

    for (int i = 0; i < logn; ++i) {
        circ << H(qubits[num_q - 1 - (i + 1)]);
    }

    circ << OA;

    for (int i = 0; i < logn; ++i) {
        circ << SWAP(qubits[num_q - 1 - (i + 1)], qubits[num_q - 1 - (i + logn + 1)]);
    }

    for (int i = 0; i < logn; ++i) {
        circ << H(qubits[num_q - 1 - (i + 1)]);
    }
    return make_pair(circ, alpha);
}


MatrixXcd block_encoding_method(MatrixXcd A) {
    auto result = fable(A);
    QCircuit circ = result.first;
    double alpha = result.second;
    // 生成量子电路的矩阵表示
    auto qvm = initQuantumMachine(QMachineType::CPU);
    auto prog = QProg();
    prog << circ;
    // std::string text_picture = draw_qprog(prog);
    // std::cout << text_picture << std::endl;
    auto matrix = getCircuitMatrix(prog, qvm);
    int num_qubits = log2(sqrt(matrix.size()));
    int num_rows = pow(2, num_qubits);
    MatrixXcd matrix_eigen = MatrixXcd::Zero(num_rows, num_rows);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_rows; j++) {
            matrix_eigen(i, j) = matrix[i*num_rows+j];
        }
    }
    destroyQuantumMachine(qvm);
    // cout << matrix_eigen << endl;
    // cout << matrix_eigen.size() << endl;
    return matrix_eigen;
}

// int main() {
//     int num_q = 2;
//     MatrixXcd A(4, 4);
//     A << complex<double>(1, 0), complex<double>(-0.5, 0),complex<double>(-0.5, 0), complex<double>(-0.5, 0),
//          complex<double>(-0.5, 0), complex<double>(0.25, 0),complex<double>(1, 0), complex<double>(0.5, 0),
//          complex<double>(-0.5, 0), complex<double>(1, 0),complex<double>(0.5, 0), complex<double>(-0.5, 0),
//          complex<double>(-0.5, 0), complex<double>(0.5, 0),complex<double>(-0.5, 0), complex<double>(0.5, 0)
//          ;
//     cout << "Matrix A:" << endl << A << endl;
//     auto result = fable(A);
//     QCircuit circ = result.first;
//     double alpha = result.second;
//     cout << "Alpha: " << alpha << endl;
//     cout << "Matrix A normalized:" << endl << A / alpha / 2 << endl;

//     // 生成量子电路的矩阵表示
//     auto qvm = initQuantumMachine(QMachineType::CPU);
//     auto prog = QProg();
//     prog << circ;
//     std::string text_picture = draw_qprog(prog);
//     std::cout << text_picture << std::endl;
//     auto matrix = getCircuitMatrix(prog, qvm);
//     cout << "Unitary matrix (top left 4x4):" << endl;
//     cout << matrix << endl;
//     destroyQuantumMachine(qvm);
//     return 0;
// }
