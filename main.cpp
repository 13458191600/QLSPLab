#include "QPanda.h"
#include "QAlg/Encode/Encode.h"
USING_QPANDA
int main()
{
    // 构建全振幅虚拟机
    auto qvm = new CPUQVM();
    qvm->init();
    std::vector<double>data{0.305508,0.305508,0.3972461,0.291738,0.305508,0.305508,0.3972461,0.291738};
    QProg prog;
    auto q = qvm->qAllocMany((int)data.size());
    vector<double> abs_alphas;
    reverse(q.begin(), q.end());
    double Lambda = accumulate(data.begin(), data.end(), 0.0);
    for (int i = 0; i < (int)data.size(); ++i) {
        abs_alphas.push_back(sqrt(1.0/Lambda*(data[i])));
    }
    for (int i = 0; i < (int)data.size(); ++i) {
        cout << abs_alphas[i] << " ";
    }
    cout << endl;
    //实例化Encode类，并调用angle_encode和dense_angle_encode接口
    Encode encode_b;
    encode_b.amplitude_encode(q, abs_alphas);
    //encode_b.dense_angle_encode(q, data);
    prog << encode_b.get_circuit();
    std::string text_picture = draw_qprog(prog);
    std::cout << text_picture << std::endl;
    //进行概率测量
    // auto result = qvm->probRunDict(prog,encode_b.get_out_qubits(),-1);
    // for (auto val : result) {
    //     std::cout << val.first <<':'<< val.second<< std::endl;
    // }
    QStat cir_matrix = getCircuitMatrix(prog);
    for (int i = 0; i < sqrt(cir_matrix.size()); i++) {
        cout << cir_matrix[i*sqrt(cir_matrix.size())]<< ",";
    }
    cout << endl;
    // 打印矩阵信息
    // std::cout << cir_matrix << std::endl;
    destroyQuantumMachine(qvm);
    return 0;

}