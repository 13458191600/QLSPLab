#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <unordered_map>
#include "QPanda.h"
#include "QAlg/Encode/Encode.h"
USING_QPANDA

using namespace std;
using namespace Eigen;
// using namespace QPanda;


// typedef Matrix<complex<float>, Dynamic, Dynamic> ComplexMatrix;
typedef Matrix<std::complex<double>, Dynamic, Dynamic> ComplexMatrix;
typedef unordered_map<string, complex<float>> PauliCoeMap;

ComplexMatrix sigmaI = ComplexMatrix::Identity(2, 2);
ComplexMatrix sigmaX(2, 2);
ComplexMatrix sigmaY(2, 2);
ComplexMatrix sigmaZ(2, 2);

void initialize_matrices() {
    sigmaX << 0, 1,
              1, 0;

    sigmaY << 0, complex<float>(0, -1),
              complex<float>(0, 1), 0;

    sigmaZ << 1, 0,
              0, -1;
}

PauliCoeMap pauli_decompose(const ComplexMatrix& A) {
    int q_num = static_cast<int>(log2(A.rows()));
    if (q_num >= 14) {
        throw runtime_error("qubit number too large, memory consumption too large");
    }

    vector<ComplexMatrix> Pauli_basis = { sigmaI, sigmaX, sigmaY, sigmaZ };
    vector<int> dimensions(2 * q_num, 2);
    ComplexMatrix reshapedA = A;

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

    PauliCoeMap result;
    vector<string> keys;
    for (int i = 0; i < pow(4, q_num); ++i) {
        string key;
        int val = i;
        for (int j = 0; j < q_num; ++j) {
            int mod = val % 4;
            if (mod == 0) key += 'I';
            else if (mod == 1) key += 'X';
            else if (mod == 2) key += 'Y';
            else key += 'Z';
            val /= 4;
        }
        keys.push_back(key);
    }

    for (int i = 0; i < keys.size(); ++i) {
        result[keys[i]] = reshapedA(i) / static_cast<float>(pow(2, q_num));
    }

    return result;
}

ComplexMatrix recombination(const PauliCoeMap& pauli_coe) {
    unordered_map<char, ComplexMatrix> op_map = {
        {'I', sigmaI},
        {'X', sigmaX},
        {'Y', sigmaY},
        {'Z', sigmaZ}
    };

    int q_num = pauli_coe.begin()->first.size();
    ComplexMatrix rho = ComplexMatrix::Zero(pow(2, q_num), pow(2, q_num));

    for (const auto& pair : pauli_coe) {
        ComplexMatrix temp = ComplexMatrix::Identity(1, 1);
        for (char op : pair.first) {
            temp = kroneckerProduct(temp, op_map[op]).eval();
        }
        rho += pair.second * temp;
    }

    return rho;
}

int main() {
    initialize_matrices();

    // Example usage
    ComplexMatrix A(4, 4);
    A << complex<float>(1, 0), complex<float>(2, 0), complex<float>(3, 0), complex<float>(4, 0),
         complex<float>(5, 0), complex<float>(6, 0), complex<float>(7, 0), complex<float>(8, 0),
         complex<float>(9, 0), complex<float>(10, 0), complex<float>(11, 0), complex<float>(12, 0),
         complex<float>(13, 0), complex<float>(14, 0), complex<float>(15, 0), complex<float>(16, 0);

    PauliCoeMap pauliCoe = pauli_decompose(A);
    ComplexMatrix recombinedA = recombination(pauliCoe);

    cout << "Original matrix A:" << endl << A << endl;
    cout << "Recombined matrix A:" << endl << recombinedA << endl;

    return 0;
}
