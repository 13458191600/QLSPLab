
#include <Eigen/QR>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace Eigen;

// Kronecker product implementation using Eigen's block operations
template<typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> kroneckerProduct(const Matrix<Scalar, Dynamic, Dynamic>& A, const Matrix<Scalar, Dynamic, Dynamic>& B) {
    int rowsA = A.rows();
    int colsA = A.cols();
    int rowsB = B.rows();
    int colsB = B.cols();

    Matrix<Scalar, Dynamic, Dynamic> result(rowsA * rowsB, colsA * colsB);

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            result.block(i * rowsB, j * colsB, rowsB, colsB) = A(i, j) * B;
        }
    }

    return result;
}


using namespace Eigen;
using namespace std;

// Function to check if a matrix is Hermitian
bool isHermitian(const MatrixXcd& matrix) {
    return matrix.isApprox(matrix.adjoint());
}

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

// Define the unitary O_c
MatrixXcd O_c(const MatrixXcd& H, int s) {
    int N = H.rows();
    MatrixXcd O_c = MatrixXcd::Zero(N * s, N * s);
    for (int j = 0; j < N; ++j) {
        auto colIndices = get_colidx_of_row(H, j);
        for (int l = 0; l < colIndices.size(); ++l) {
            O_c(colIndices[l] + j * s,l + j * s) = 1.0;
        }
    }
    return O_c;
}

// Define the unitary O_A
MatrixXcd O_A(const MatrixXcd& H, int s) {
    int N = H.rows();
    MatrixXcd O_A = MatrixXcd::Zero(2 * s * N, 2 * s * N);
    for (int j = 0; j < N; ++j) {
        auto colIndices = get_colidx_of_row(H, j);
        for (int l = 0; l < colIndices.size(); ++l) {
            complex<double> A_value = H(colIndices[l], j);
            O_A(0 + l * 2 + j * 2 * s, 0 + l * 2 + j * 2 * s) = A_value;
            O_A(1 + l * 2 + j * 2 * s, 0 + l * 2 + j * 2 * s) = sqrt(1.0 - pow(norm(A_value),2));
        }
    }
    return O_A;
}

// Define the diffusion operator D_s
MatrixXcd D_s(int m) {
    MatrixXcd H(2, 2);
    H << 1/sqrt(2),1/sqrt(2),
        1/sqrt(2),-1/sqrt(2);
    if(m ==1){
        return H;
    }
    MatrixXcd D_s = kroneckerProduct(H,H);
    for(int i =0;i<m-1;i++){
        MatrixXcd D_s = kroneckerProduct(H,D_s);
    }
    return D_s;
}

// Function to construct the block encoding unitary U_A
MatrixXcd block_encoding_method(const MatrixXcd& SparseHamiltonian) {
    if (!isHermitian(SparseHamiltonian)) {
        throw invalid_argument("The Hamiltonian is not Hermitian.");
    }

    int N = SparseHamiltonian.rows();
    int m = log2(N);  // Assuming N is a power of 2
    int s = pow(2, m);
    
    MatrixXcd Oc = O_c(SparseHamiltonian, s);
    MatrixXcd Oa = O_A(SparseHamiltonian, s);
    MatrixXcd Ds = D_s(m);

    // Constructing I_2 \otimes D_s \otimes I_N
    MatrixXcd I2 = MatrixXcd::Identity(2, 2);
    MatrixXcd IN = MatrixXcd::Identity(N, N);
    MatrixXcd I2_Ds_IN = kroneckerProduct(I2, kroneckerProduct(Ds, IN));
    MatrixXcd I2_Oc = kroneckerProduct(I2, Oc);

    // Constructing the block encoding unitary U_A
    MatrixXcd Ua = I2_Ds_IN * (I2_Oc) * Oa * I2_Ds_IN;

    return Ua;
}
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
bool verify_block_encoding(const MatrixXcd& U, const MatrixXcd& H, int s) {
    // int s = pow(2, m);
    bool verified = true;
    int d_h = H.rows();
    int d_a = U.rows()/H.rows();
    MatrixXcd PU = partialTraceFirstSubsystem(U,d_a,d_h);
    for (int i = 0; i < H.rows(); ++i) {
        for (int j = 0; j < H.cols(); ++j) {
            complex<double> element = U(i * d_a, j*d_a);
            // complex<double> element = PU(i,j);
            if (abs(element - H(i, j) / double(s)) > 1e-6) {
                cout << "Verification failed at (" << i << ", " << j << "): " << element << " != " << H(i, j)/double(s)<< endl;
                verified = false;
            }
        }
    }

    return verified;
}
int main() {
    // Example of a sparse Hamiltonian H
    MatrixXcd H(2, 2);
    H << complex<double>(1.0, 0.0), complex<double>(0.5, 0.0),
         complex<double>(0.5, 0.0), complex<double>(0.0, 0.0);
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
    bool result = verify_block_encoding(U, H, 2);

    if (result) {
        cout << "Block encoding verification succeeded." << endl;
    } else {
        cout << "Block encoding verification failed." << endl;
    }
    cout << "Block Encoding Unitary U: \n" << U << endl;

    return 0;
}
