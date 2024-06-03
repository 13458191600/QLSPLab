#include <iostream>
#include <Eigen/Dense>
#include <QPanda.h>
#include <cmath>
#include <vector>
#include <complex>
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
                a[j] = (x + y) / 2;
                a[j + (1 << h)] = (x - y) / 2;
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


QCircuit compressed_uniform_rotation(const VectorXd& a, bool ry = true) {
    int n = static_cast<int>(log2(a.size()) / 2);
    auto qvm = initQuantumMachine(QMachineType::CPU);
    QCircuit circ;
    auto qubits = qvm->allocateQubits(2 * n + 1);

    int i = 0;
    while (i < a.size()) {
        int parity_check = 0;

        if (a[i] != 0) {
            if (ry) {
                circ << RY(qubits[0], a[i]);
            } else {
                circ << RZ(qubits[0], a[i]);
            }
        }

        while (true) {
            int ctrl = compute_control(i + 1, n);
            parity_check ^= (1 << (ctrl - 1));
            i++;
            if (i >= a.size() || a[i] != 0) {
                break;
            }
        }

        for (int j = 1; j <= 2 * n; ++j) {
            if (parity_check & (1 << (j - 1))) {
                circ << CNOT(qubits[j], qubits[0]);
            }
        }
    }

    return circ;
}


// FABLE
pair<QCircuit, double> fable(MatrixXd a, double epsilon = -1) {
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
        MatrixXd padded = MatrixXd::Zero(1 << logn, 1 << logn);
        padded.topLeftCorner(n, n) = a;
        a = padded;
    }

    VectorXd vec_a = Map<VectorXd>(a.data(), a.size());
    vec_a = gray_permutation(sfwht(2.0 * vec_a.array().acos().matrix()));

    if (epsilon >= 0) {
        for (int i = 0; i < vec_a.size(); ++i) {
            if (abs(vec_a[i]) <= epsilon) {
                vec_a[i] = 0;
            }
        }
    }

    QCircuit OA = compressed_uniform_rotation(vec_a);
    QCircuit circ;
    auto qvm = initQuantumMachine(QMachineType::CPU);
    auto qubits = qvm->allocateQubits(2 * logn + 1);

    for (int i = 0; i < logn; ++i) {
        circ << H(qubits[i + 1]);
    }

    circ << OA;

    for (int i = 0; i < logn; ++i) {
        circ << SWAP(qubits[i + 1], qubits[i + logn + 1]);
    }

    for (int i = 0; i < logn; ++i) {
        circ << H(qubits[i + 1]);
    }

    circ << circ.reverse_bits();

    return make_pair(circ, alpha);
}


int main() {
    MatrixXd A = Random(4, 4);
    A += A.transpose();
    cout << "Matrix A:" << endl << A << endl;

    auto result = fable(A);
    QCircuit circ = result.first;
    double alpha = result.second;
    cout << "Alpha: " << alpha << endl;
    cout << "Matrix A normalized:" << endl << A / alpha / 4 << endl;

    // 生成量子电路的矩阵表示
    auto qvm = initQuantumMachine(QMachineType::CPU);
    auto prog = QProg();
    prog << circ;
    auto matrix = getCircuitMatrix(prog, qvm);
    cout << "Unitary matrix (top left 4x4):" << endl;
    cout << matrix.topLeftCorner(4, 4) << endl;

    // 打印差值
    cout << "Difference:" << endl;
    cout << matrix.topLeftCorner(4, 4) - A / alpha / 4 << endl;

    destroyQuantumMachine(qvm);
    return 0;
}
