#include "QPanda.h"
#include <Eigen/Dense>
#include <iostream>
#include <complex>
#include <limits>
USING_QPANDA

using namespace Eigen;
using namespace std;
typedef complex<double> Complex;

void phase_encode(QCircuit& qc, QVec& qvec, vector<double>& phase) {

    QMatrixXcd P = QMatrixXcd::Zero(phase.size(), phase.size());
    for (int i = 0; i < phase.size(); i++) {
        P(i, i) = complex<double>(cos(phase[i]), sin(phase[i]));
    }
    int num_qubits = log2(P.rows());
    
    QVec qvec_tmp = QVec(qvec.begin() , qvec.begin()+num_qubits);
    reverse(qvec_tmp.begin(), qvec_tmp.end());
    qc << matrix_decompose_qr(qvec_tmp, P);

}

unsigned int reverseBits(unsigned int num, int bitLength) {
    bitset<32> bits(num); // 使用bitset处理32位整数
    bitset<32> reversedBits;
    // 反转比特串
    for (int i = 0; i < bitLength; ++i) {
        reversedBits[bitLength - 1 - i] = bits[i];
    }
    // 将反转后的比特串转换为整数
    return reversedBits.to_ulong();
}

void initialize(QProg& qc, QVec& qvec, VectorXcd bvec) {
    // check the qvec size align with the bvec size
    if (pow(2, qvec.size()) != bvec.size()) {
        cout << "Error: qubit size" << qvec.size() << " and initial state size" << bvec.size() << " not align!" << endl;
        char format_str[64] = { 0 };
        snprintf(format_str, sizeof(format_str) - 1, "qubit size with %d and initial state qubit size %d  not align!", (int)qvec.size(), (int)log2(bvec.size()));
        throw length_error(format_str);
        return;
    }
    Encode encode_b;
    double epsm = numeric_limits<double>::epsilon();
    int isreal = 1;
    reverse(qvec.begin(), qvec.end());
    // check the data is all real
    for (int i = 0; i < bvec.size(); i++) {
        if (bvec[i].imag() > epsm){
            isreal = 0;
            break;
        }
    }
    vector<long double> bamp(bvec.size());
    vector<long double> bphase(bvec.size());
    if (isreal) {
        for (int i = 0; i < bvec.size(); i++) {
            bamp[i] = bvec[i].real();
        }
        long double norm = 0;
        for (int i = 0; i < bamp.size(); i++) {
            norm += (bamp[i]) *(bamp[i]);
        }
        norm = sqrt(norm);
        vector<double> bampdouble(bvec.size());
        // cout << "The amplitude of initial state is: " << norm << endl;
        for (int i = 0; i < bamp.size(); i++) {
            bampdouble[i] = bamp[i]/norm;
        }
        encode_b.amplitude_encode(qvec, bampdouble);
        qc << encode_b.get_circuit();
    }else{
        for (int i = 0; i < bvec.size(); i++) {
            bamp[i] = abs(bvec[i]);
            bphase[i] = arg(bvec[i]);
        }
        long double norm = 0;
        for (int i = 0; i < bamp.size(); i++) {
            norm += (bamp[i]) *(bamp[i]);
        }
        norm = sqrt(norm);
        vector<double> bampdouble(bvec.size());
        for (int i = 0; i < bamp.size(); i++) {
            bampdouble[i] = bamp[i]/norm;
        }
        if (abs(pow(norm, 2) - 1) > 1e-10) {
            cout << "Error: the amplitude of initial state is not normalized!" << endl;
            char format_str[64] = { 0 };
            cout << fixed << setprecision(32);
            cout << "norm is " << norm << endl;
            snprintf(format_str, sizeof(format_str) - 1, "the amplitude of initial state is not normalized norm is %.32f!", epsm);
            throw length_error(format_str);
            return;
        }
        encode_b.amplitude_encode(qvec, bampdouble);
        qc << encode_b.get_circuit();

        QMatrixXcd P = QMatrixXcd::Zero(bphase.size(), bphase.size());
        for (int i = 0; i < bphase.size(); i++) {
            // reverse the bits of the phase
            P(i,i) = complex<double>(cos(bphase[i]), sin(bphase[i]));
        }
        qc << matrix_decompose_qr(qvec, P);
    }
    reverse(qvec.begin(), qvec.end());
}

VectorXcd getQuantumStates( QuantumMachine * qvm, QProg& prog) {
    qvm->directlyRun(prog);
    QStat state = qvm->getQState(); // Get the quantum state
    unsigned int bitLength = log2(state.size());
    VectorXcd result(state.size());
    for (int i = 0; i < state.size(); i++) {
        result[i] = state[reverseBits(i, bitLength)];
    }
    return result;
}
void printCircuit(QProg& prog) {
    string text_picture = draw_qprog(prog);
    cout << text_picture << endl;
}

void printUnitary(QProg& prog) {
    QStat cir_matrix = getCircuitMatrix(prog);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cout << cir_matrix[i*sqrt(cir_matrix.size())+j]<< ",";
        }
    }
    cout << endl;
    // 打印矩阵信息
    cout << cir_matrix << endl;
}


// int main() {
//     auto qvm = new CPUQVM();
//     qvm->init();
//     QVec qvec = qvm->qAllocMany(3);
//     VectorXcd iniState = VectorXcd::Random(pow(2,2));
//     iniState.normalize();
//     cout << "The initial state is: " << iniState << endl;
//     QProg cir;
//     QVec qvec_tmp = QVec(qvec.begin(), qvec.begin()+2);
//     initialize(cir, qvec, iniState);
//     cir << X(qvec[2]) << X(qvec[2]);
//     QProg prog;
//     prog << cir;
//     string text_picture = draw_qprog(prog);
//     cout << text_picture << endl;
//     //进行概率测量
//      // Run the program on the quantum virtual machine
//     VectorXcd state_real = getQuantumStates(qvm, prog);
//     cout << "The quantum state is: " << state_real << endl;
//     destroyQuantumMachine(qvm);
//     return 0;
// }