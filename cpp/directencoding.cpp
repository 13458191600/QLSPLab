#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

// Function to get the set of column indices of all non-zero elements in row j
vector<int> get_colidx_of_row(const MatrixXcd& SparseHamiltonian, int j) {
    vector<int> colIndices;
    for (int k = 0; k < SparseHamiltonian.cols(); ++k) {
        if (SparseHamiltonian(j, k) != complex<double>(0.0, 0.0)) {
            colIndices.push_back(k);
        }
    }
    return colIndices;
}

// Function to construct the isometry T1
MatrixXcd construct_T1(const MatrixXcd& H, double H_max) {
    int d = H.cols();
    MatrixXcd T1 = MatrixXcd::Zero(1,d*d*2*2);

    for (int j = 0; j < H.rows(); ++j) {
        auto F_j = get_colidx_of_row(H, j);
        for (int p : F_j) {
            T1(0,0+0*2+p*2*2+j*d*2*2) = sqrt(H(p, j) / H_max) / sqrt(d);
            T1(0,1+0*2+p*2*2+j*d*2*2) = sqrt(1 - abs(H(p, j) / H_max)) / sqrt(d);
        }
    }

    return T1;
}

// Function to construct the isometry T2
MatrixXcd construct_T2(const MatrixXcd& H, double H_max) {
    int d = H.cols();
    MatrixXcd T2 = MatrixXcd::Zero(1,d*d*2*2);

    for (int k = 0; k < H.rows(); ++k) {
        auto F_k = get_colidx_of_row(H, k);
        for (int p : F_k) {
            T2(0,0+0*2+k*2*2+p*d*2*2) = sqrt(H(p,k) / H_max) / sqrt(d);
            T2(0,0+1*2+k*2*2+p*d*2*2) = sqrt(1 - abs(H(p,k) / H_max)) / sqrt(d);
        }
    }

    return T2;
}

bool isHermitian(const MatrixXcd& matrix) {
    return matrix.isApprox(matrix.adjoint());
}

// Function to construct the block encoding unitary U
MatrixXcd block_encoding_method(const MatrixXcd& SparseHamiltonian) {
    double H_max = SparseHamiltonian.cwiseAbs().maxCoeff();
    MatrixXcd T1 = construct_T1(SparseHamiltonian, H_max);
    MatrixXcd T2 = construct_T2(SparseHamiltonian, H_max);

    // U = T2^dagger * T1
    MatrixXcd U = T2.adjoint() * T1;

    return U;
}

// Function to compute the partial trace over the second subsystem
MatrixXcd partialTraceSecondSubsystem(const MatrixXcd& rho, int d_A, int d_B) {
    // The dimension of the composite system
    int d = d_A * d_B;
    
    // Initialize the reduced density matrix
    MatrixXcd rho_A = MatrixXcd::Zero(d_A, d_A);
    
    // Perform the partial trace
    for (int i = 0; i < d_B; ++i) {
        for (int j = 0; j < d_B; ++j) {
            // Extract the block corresponding to (i, j) in the second subsystem
            MatrixXcd block = rho.block(i * d_A, j * d_A, d_A, d_A);
            
            // Add the block to the reduced density matrix if i == j
            if (i == j) {
                rho_A += block;
            }
        }
    }
    
    return rho_A;
}
// Function to compute the partial trace over the first subsystem
MatrixXcd partialTraceFirstSubsystem(const MatrixXcd& rho, int d_A, int d_B) {
    // The dimension of the composite system
    int d = d_A * d_B;

    // Initialize the reduced density matrix
    MatrixXcd rho_B = MatrixXcd::Zero(d_B, d_B);

    // Perform the partial trace
    for (int i = 0; i < d_A; ++i) {
        for (int j = 0; j < d_A; ++j) {
            if (i == j) {
                // Extract the block corresponding to (i, j) in the first subsystem
                MatrixXcd block = rho.block(i * d_B, j * d_B, d_B, d_B);
                // Add the block to the reduced density matrix if i == j
                rho_B += block;
            }
        }
    }

    return rho_B;
}
// Function to verify the block encoding
bool verify_block_encoding(const MatrixXcd& U, const MatrixXcd& H) {
    int d = H.rows();
    int d_A = pow(2,log2(U.rows())- log2(d));
    bool verified = true;

    MatrixXcd partialU = partialTraceFirstSubsystem(U,d_A,d);
    for (int i = 0; i < H.rows(); ++i) {
        for (int j = 0; j < H.cols(); ++j) {
            complex<double> element = partialU(i , j);
            if (abs(element - H(i, j) / double(d)) > 1e-6) {
                cout << "Verification failed at (" << i << ", " << j << "): " << element << " != " << H(i, j) / double(d) << endl;
                verified = false;
            }
        }
    }

    return verified;
}

int main() {
    // Example of a sparse Hamiltonian H
    MatrixXcd H(2, 2);
    H << complex<double>(1.0, 0.0), complex<double>(1, 0.0),
         complex<double>(1, 0.0), complex<double>(0.0, 0.0);
    // H << complex<double>(1.0, 0.0), complex<double>(0.5, 0.5), complex<double>(0.0, 0.0), complex<double>(0.0, 0.0),
    //      complex<double>(0.5, -0.5), complex<double>(1.0, 0.0), complex<double>(0.5, 0.5), complex<double>(0.0, 0.0),
    //      complex<double>(0.0, 0.0), complex<double>(0.5, -0.5), complex<double>(1.0, 0.0), complex<double>(0.5, 0.5),
    //      complex<double>(0.0, 0.0), complex<double>(0.0, 0.0), complex<double>(0.5, -0.5), complex<double>(1.0, 0.0);

    // Ensure the matrix is Hermitian
    if (!isHermitian(H)) {
        cout << "The Hamiltonian is not Hermitian." << endl;
        return -1;
    }

    MatrixXcd U = block_encoding_method(H);
    bool result = verify_block_encoding(U, H);

    cout << "Block Encoding Unitary U: \n" << U << endl;

    return 0;
}
