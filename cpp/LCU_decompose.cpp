#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <map>
#include <cmath>
#include <unordered_map>
#include "QPanda.h"
#include "QAlg/Encode/Encode.h"
USING_QPANDA

using namespace std;
using namespace Eigen;
using namespace QPanda;

typedef map<string, complex<double>> PauliCoeMap;
// Define Pauli matrices
Matrix2cd pauliI() {
    Matrix2cd I;
    I << 1, 0, 0, 1;
    return I;
}

Matrix2cd pauliX() {
    Matrix2cd X;
    X << 0, 1, 1, 0;
    return X;
}

Matrix2cd pauliY() {
    Matrix2cd Y;
    Y << 0, complex<double>(0, -1), complex<double>(0, 1), 0;
    return Y;
}
Matrix2cd pauliZ() {
    Matrix2cd Z;
    Z << 1, 0, 0, -1;
    return Z;
}
// MatrixXcd kroneckerProduct(const MatrixXcd &A, const MatrixXcd &B) {
//     MatrixXcd result(A.rows() * B.rows(), A.cols() * B.cols());
//     for (int i = 0; i < A.rows(); ++i) {
//         for (int j = 0; j < A.cols(); ++j) {
//             result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
//         }
//     }
//     return result;
// }

// Generate tensor products of Pauli matrices
pair<vector<string>,vector<MatrixXcd>> generatePauliBasis(int n) {
    vector<Matrix2cd> paulis = {pauliI(), pauliX(), pauliY(), pauliZ()};
    char keys[4] = {'I', 'X', 'Y', 'Z'};
    vector<MatrixXcd> basis;
    vector<string> paulisStr;
    int total = pow(4, n);
    for (int i = 0; i < total; ++i) {
        MatrixXcd tensorProduct = MatrixXcd::Identity(1, 1);
        int idx = i;
        cout << i << " -> ";
        string paulistr = "";
        for (int j = 0; j < n; ++j) {
            tensorProduct = kroneckerProduct(tensorProduct, paulis[idx % 4]).eval();
            // cout << idx % 4;
            paulistr += keys[idx % 4];
            idx /= 4;
        }
        cout << paulistr << endl;
        paulisStr.push_back(paulistr);
        basis.push_back(tensorProduct);
    }
    return {paulisStr, basis};
}



// Function to calculate trace
complex<double> trace(const MatrixXcd &matrix) {
    return matrix.trace();
}

// Function to decompose matrix A using Pauli bases
PauliCoeMap pauli_decompose(const MatrixXcd &A) {
    vector<complex<double>> coefficients;
    int n = log2(A.rows());
    auto [paulisStr, basis] = generatePauliBasis(n);
    int total = pow(4, n);
    PauliCoeMap result;
    for (int i = 0; i < total; ++i) {
        complex<double> coefficient = (1.0 / pow(2, n)) * trace(A * basis[i]);
        if (abs(coefficient) < 1e-10) continue;
        result[paulisStr[i]] = coefficient;
    }
    return result;
}

pair<bool, int> judge_2_powers(int k) {
    if (k <= 0) return {false, 0};
    int log2_k = log2(k);
    bool is_power_of_2 = (1 << log2_k) == k;
    return {is_power_of_2, log2_k};
}

QProg block_encoding_LCU_pauli(const MatrixXcd& A) {
    auto [is_power_of_2, qnum_encode] = judge_2_powers(A.rows());
    if (!is_power_of_2) {
        throw runtime_error("Matrix dimension is not a power of 2.");
    }
    PauliCoeMap alphas = pauli_decompose(A);
    // get the size of alphas
    int M = alphas.size();

    // get the number of qubits needed to encode the matrix
    int m = log2(M);
    int qnum_ancilla;
    if (M != pow(2, m))
        qnum_ancilla = m+1;
    else
        qnum_ancilla = m;

    // create a quantum circuit with qnum_all qubits
    int qnum_all = qnum_encode + qnum_ancilla;

    vector<double> abs_alphas;
    for (const auto& [key, alpha] : alphas) {
        cout << key << " : " << alpha << endl;
        abs_alphas.push_back(fabs<double>(alpha));
    }

    double Lambda = accumulate(abs_alphas.begin(), abs_alphas.end(), 0.0);
    for (int i = M; i < pow(2, qnum_ancilla); ++i) {
        abs_alphas.push_back(0.0);
    }
    for (double& init_state : abs_alphas) {
        init_state = sqrt(init_state / Lambda);
    }
    // cout abs_alphas
    cout << "Lambda: " << Lambda << endl;
    for (int i = 0; i < abs_alphas.size(); ++i) {
        cout << "|" << i << "> : " << abs_alphas[i];
    }
    cout << endl;


    // create a quantum circuit to initialize the state
    auto qvm = new CPUQVM();
    qvm->init();
    auto qubits = qvm->qAllocMany((int)abs_alphas.size());
    // reverse(qubits.begin(), qubits.end());
    auto encode_qubits = vector<Qubit*>(qubits.begin() , qubits.begin()+qnum_encode);
    // reverse(encode_qubits.begin(), encode_qubits.end());
    auto ancillas = vector<Qubit*>(qubits.begin()+qnum_encode, qubits.begin()+qnum_encode+qnum_ancilla);
    // reverse(ancillas.begin(), ancillas.end());
    // encode the alphas by amplitude encoding
    Encode encode_b;
    encode_b.amplitude_encode(ancillas, abs_alphas);
    QCircuit qc_init;
    qc_init << encode_b.get_circuit();  // Placeholder for state initialization

    // Todo: 这里需要反转一下量子比特顺序,因为这个编码生成的比特顺序是反的
    
    QProg qc_all;
    qc_all << qc_init;
    int idx1 = 0;
    for (const auto& pauli_basis : alphas) {
        complex<double> coe = pauli_basis.second;
        // add X gate to specify the control value
        // get the bit string of idx1
        const size_t bit_len = qnum_ancilla;
        string bit_str = bitset<32>(idx1).to_string();
        // clip the bit string to the same length as the pauli_basis
        bit_str = bit_str.substr(32- bit_len);
        cout<< bit_str << endl;
        // reverse(bit_str.begin(), bit_str.end());
        for (size_t idx2 = 0; idx2 < qnum_ancilla; ++idx2) {
            if (bit_str[idx2] == '0') {
                qc_all << X(ancillas[idx2]);
            }
        }
        double angle = arg(coe);
        for (size_t idx2 = 0; idx2 < pauli_basis.first.size(); ++idx2) {
            char _pauli_basis = pauli_basis.first[idx2];
            if (idx2 == 0) {
                // a global phase gate
                qc_all << U1(encode_qubits[idx2],angle).control(ancillas);
                qc_all << X(encode_qubits[idx2]).control(ancillas);
                qc_all << U1(encode_qubits[idx2],angle).control(ancillas);
                qc_all << X(encode_qubits[idx2]).control(ancillas);
            }
            if (_pauli_basis == 'X') {
                qc_all << X(encode_qubits[idx2]).control(ancillas);
            } else if (_pauli_basis == 'Y') {
                qc_all << Y(encode_qubits[idx2]).control(ancillas);
            } else if (_pauli_basis == 'Z') {
                qc_all << Z(encode_qubits[idx2]).control(ancillas);
            }
            
        }
        for (size_t idx2 = 0; idx2 < qnum_ancilla; ++idx2) {
            if (bit_str[idx2] == '0') {
                qc_all << X(ancillas[idx2]);
            }
        }
        idx1++;
    }

    qc_all << qc_init.dagger();
    // // destroyQuantumMachine(qvm);
    // return qc_all;
    return qc_all;
}

int main() {

    // Define test matrix A
    MatrixXcd A(2, 2);
    A << complex<double>(0.5, 0), complex<double>(-0.8, 0), 
         complex<double>(-0.8, 0), complex<double>(0.5, 0);

    QProg qc_all = block_encoding_LCU_pauli(A);

    // Output results
    cout << "Matrix A:" << endl << A << endl;
    // Further code to simulate the quantum circuit and output results
    string text_picture = draw_qprog(qc_all);

    cout << text_picture << endl;

    // Simulate the quantum circuit
    QStat cir_matrix = getCircuitMatrix(qc_all);
    cout << fixed << setprecision(2);
    int row = sqrt(cir_matrix.size());
    cout << "Circuit matrix:" << endl;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            cout << cir_matrix[i*row+j] << " ";
        }
        cout << endl;
    }
}


