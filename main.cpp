#include "QPanda.h"
USING_QPANDA

int main()
{
    // 初始化量子虚拟机
    init(QMachineType::CPU);

    // 申请量子比特以及经典寄存器
    auto q = qAllocMany(2);
    auto c = cAllocMany(2);

    // 构建量子程序
    QProg prog;
    prog << H(q[0])
        << CNOT(q[0],q[1])
        << MeasureAll(q, c);

    // 量子程序运行1000次，并返回测量结果
    auto results = runWithConfiguration(prog, c, 1000);

    // 打印量子态在量子程序多次运行结果中出现的次数
    for (auto &val: results)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }

    // 释放量子虚拟机
    finalize();

    return 0;
}