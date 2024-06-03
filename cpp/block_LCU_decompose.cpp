#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <map>
#include <unordered_map>
#include "QPanda.h"
#include "QAlg/Encode/Encode.h"
USING_QPANDA

using namespace std;
using namespace Eigen;
using namespace QPanda;

typedef Matrix<std::complex<double>, Dynamic, Dynamic> ComplexMatrix;
typedef map<char, ComplexMatrix> PauliCoeMap;

ComplexMatrix sigmaI = ComplexMatrix::Identity(2, 2);
ComplexMatrix sigmaX(2, 2); 
ComplexMatrix sigmaY(2, 2);
ComplexMatrix sigmaZ(2, 2);

void initialize_matrices() {
    sigmaX << 0, 1,
              1, 0;

    sigmaY << 0, std::complex<double>(0, -1),
              std::complex<double>(0, 1), 0;

    sigmaZ << 1, 0,
              0, -1;
}

map<char, ComplexMatrix>  pauli_decompose(const ComplexMatrix& A) {
    int q_num = static_cast<int>(log2(A.rows()));
    if (q_num >= 14) {
        throw runtime_error("qubit number too large, memory consumption too large");
    }

    vector<ComplexMatrix> Pauli_basis = { sigmaI, sigmaX, sigmaY, sigmaZ };
    // vector<int> dimensions(2 * q_num, 2);
    ComplexMatrix reshapedA = A; // A is a 2^q_num x 2^q_num matrix

    for (int i = 0; i < q_num; ++i) {
        reshapedA = reshapedA.transpose();
    }

    for (int i = 0; i < q_num; ++i) {
        ComplexMatrix temp(4, reshapedA.cols() / 2);
        for (int j = 0; j < 4; ++j) {
            temp.row(j) = reshapedA * Pauli_basis[j];
        }
        reshapedA = temp;
    }

    map<char, ComplexMatrix> result;
    char keys[4] = {"I", "X", "Y", "Z"};
    for (int i = 0; i < keys.size(); ++i) {
        result[keys[i]] = reshapedA(i) / static_cast<complex>(pow(2, q_num));
    }

    return result;
}

pair<bool, int> judge_2_powers(int k) {
    if (k <= 0) return {false, 0};
    int log2_k = log2(k);
    bool is_power_of_2 = (1 << log2_k) == k;
    return {is_power_of_2, log2_k};
}

QuantumCircuit block_encoding_LCU_pauli(const ComplexMatrix& A) {
    auto [is_power_of_2, qnum_encode] = judge_2_powers(A.rows());
    if (!is_power_of_2) {
        throw runtime_error("Matrix dimension is not a power of 2.");
    }

    int qnum_ancilla = 2 * qnum_encode;
    int qnum_all = qnum_encode + qnum_ancilla;

    PauliCoeMap alphas = pauli_decompose(A);
    vector<double> abs_alphas;
    for (const auto& [key, init_state] : alphas) {
        abs_alphas.push_back(abs(init_state));
    }

    double Lambda = accumulate(abs_alphas.begin(), abs_alphas.end(), 0.0);
    for (double& init_state : abs_alphas) {
        init_state = sqrt(init_state / Lambda);
    }
    
    auto qvm = CPUQVM();
    qvm.init();
    // # encode to ciecuit
    auto qubits_encode = qvm.qAllocMany(qnum_encode);
    Encode encode_state;
    encode_state.basic_encode(qnum_encode,encode_state);
    encode_circuit = encode_state.get_circuit();

    auto qubits = qvm.qAllocMany(qnum_all);
    auto ancillas = vector<Qubit*>(qubits.begin(), qubits.begin() + qnum_ancilla);
    auto encode_qubits = vector<Qubit*>(qubits.begin() + qnum_ancilla, qubits.end());

    // QuantumCircuit qc_init;
    QCircuit qc_init = QCircuit();
    qc_init << H(ancillas);  // Placeholder for state initialization

    QCircuit qc_all = QCircuit();
    qc_all << qc_init;

    for (const auto& [idx1, pauli_basis] : enumerate(alphas)) {
        complex<double> coe = pauli_basis.second;
        if (abs(coe) == 0) continue;
        for (size_t idx2 = 0; idx2 < pauli_basis.first.size(); ++idx2) {
            char _pauli_basis = pauli_basis.first[idx2];
            QGate gate;
            if (_pauli_basis == 'I') {
                gate = I(encode_qubits[idx2]);
            } else if (_pauli_basis == 'X') {
                gate = X(encode_qubits[idx2]);
            } else if (_pauli_basis == 'Y') {
                gate = Y(encode_qubits[idx2]);
            } else if (_pauli_basis == 'Z') {
                gate = Z(encode_qubits[idx2]);
            }
            qc_all << gate.control(ancillas);
        }
    }

    qc_all << qc_init.dagger();
    finalizeQuantumMachine(qvm);
    return qc_all;
}

int main() {
    initialize_matrices();

    // Define test matrix A
    ComplexMatrix A(4, 4);
    A << complex<double>(0.5, 0), complex<double>(-0.5, 0), complex<double>(0.5, 0), complex<double>(-0.5, 0),
         complex<double>(0.5, 0), complex<double>(-0.5, 0), complex<double>(0.5, 0), complex<double>(0.5, 0),
         complex<double>(0.5, 0), complex<double>(-0.5, 0), complex<double>(0.5, 0), complex<double>(-0.5, 0),
         complex<double>(0.5, 0), complex<double>(-0.5, 0), complex<double>(0.5, 0), complex<double>(0.5, 0);

    QCircuit qc_all = block_encoding_LCU_pauli(A);

    // Output results
    cout << "Matrix A:" << endl << A << endl;
    // Further code to simulate the quantum circuit and output results

    return 0;
}
