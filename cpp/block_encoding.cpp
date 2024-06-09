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

VectorXd gray_permutation(const VectorXd& a) {
    VectorXd b = VectorXd::Zero(a.size());
    for (int i = 0; i < a.size(); ++i) {
        b[i] = a[gray_code(i)];
    }
    return b;
}

VectorXd sfwht(VectorXd a) {
    int n = static_cast<int>(log2(a.size()));
    for (int h = 0; h < n; ++h) {
        for (int i = 0; i < a.size(); i += (1 << (h + 1))) {
            for (int j = i; j < i + (1 << h); ++j) {
                double x = a[j];
                double y = a[j + (1 << h)];
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
    auto qubits = qvm->allocateQubits(num_q);

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
    int m = a.cols();
    if (n != m) {
        int max_dim = max(n, m);
        MatrixXcd padded = MatrixXcd::Zero(max_dim, max_dim);
        padded.topLeftCorner(n, m) = a;
        a = padded;
    }

    // cout << "Flat A:" << endl << flat_a.transpose() << endl;
    int logn = static_cast<int>(ceil(log2(n)));
    if (n < (1 << logn)) {
        MatrixXcd padded = MatrixXcd::Zero(1 << logn, 1 << logn);
        padded.topLeftCorner(n, n) = a;
        a = padded;
    }
    // flatten A
    VectorXcd flat_a = Map<VectorXcd>(a.data(), a.size());
    // check if A is real or complex
    int isreal = 1;
    for (int i = 0; i < flat_a.size(); ++i) {
        if (abs(imag(flat_a[i])) > epsm) {
            isreal = 0;
            break;
        }
    }
    QCircuit OAm;
    QCircuit OAp;
    if(isreal) {
        // Real data
        VectorXd real_a(flat_a.size());
        // transform to angle
        for (int i = 0; i < flat_a.size(); ++i) {
            real_a[i] = acos(real(flat_a[i])) * 2.0;
        }
        flat_a = gray_permutation(sfwht(real_a));
        if (epsilon > 0) {
            for (int i = 0; i < flat_a.size(); ++i) {
                if (abs(flat_a[i]) <= epsilon) {
                    flat_a[i] = complex<double>(0.0, 0.0);
                }
            }
        }
        OAm << compressed_uniform_rotation(flat_a);
    } else {
        // Complex data
        VectorXd a_m(flat_a.size()), a_p(flat_a.size());
        // transform to angle and phase

        for (int i = 0; i < flat_a.size(); ++i) {
            a_m[i] = acos(abs(flat_a[i])) * 2.0;
            a_p[i] = arg(flat_a[i]) * -2.0;
        }
        a_m = gray_permutation(sfwht(a_m));
        a_p = gray_permutation(sfwht(a_p));

        if (epsilon > 0) {
            for (int i = 0; i < a_m.size(); ++i) {
                if (abs(a_m[i]) <= epsilon) {
                    a_m[i] = 0.0;
                }
                if (abs(a_p[i]) <= epsilon) {
                    a_p[i] = 0.0;
                }
            }
        }

        OAm << compressed_uniform_rotation(a_m);
        OAp << compressed_uniform_rotation(a_p, false);
    }
    QCircuit circ;
    auto qvm = new CPUQVM();
    qvm->init();
    int num_q = 2 * logn + 1;
    auto qubits = qvm->qAllocMany(2 * logn + 1);

    for (int i = 0; i < logn; ++i) {
        circ << H(qubits[num_q - 1 - (i + 1)]);
    }
    if (isreal) {
        circ << OAm;

    } else {
        circ << OAm << OAp;

    }

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
    // string text_picture = draw_qprog(prog);
    // cout << text_picture << endl;
    auto matrix = getCircuitMatrix(prog, qvm);
    int num_qubits = log2(sqrt(matrix.size()));
    int num_rows = pow(2, num_qubits);
    MatrixXcd matrix_eigen = MatrixXcd::Zero(num_rows, num_rows);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_rows; j++) {
            matrix_eigen(i, j) = matrix[i+j*num_rows];
        }
    }
    destroyQuantumMachine(qvm);
    return matrix_eigen;
}


pair<MatrixXcd,QCircuit> block_encoding_circuit(MatrixXcd A) {
    auto result = fable(A);
    QCircuit circ = result.first;
    double alpha = result.second;
    // 生成量子电路的矩阵表示
    auto qvm = initQuantumMachine(QMachineType::CPU);
    auto prog = QProg();
    prog << circ;
    string text_picture = draw_qprog(prog);
    cout << text_picture << endl;
    auto matrix = getCircuitMatrix(prog, qvm);
    int num_qubits = log2(sqrt(matrix.size()));
    int num_rows = pow(2, num_qubits);
    MatrixXcd matrix_eigen = MatrixXcd::Zero(num_rows, num_rows);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_rows; j++) {
            matrix_eigen(i, j) = matrix[i+j*num_rows];
        }
    }
    destroyQuantumMachine(qvm);
    // cout << matrix_eigen << endl;
    // cout << matrix_eigen.size() << endl;
    return {matrix_eigen,circ};
}

