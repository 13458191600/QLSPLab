## 环境搭建
### 安装Qpanda Linux C++ 版
首先下载源码
```bash
git clone https://gitee.com/OriginQ/QPanda-2.git
```
然后进入QPanda-2/
```bash
cd QPanda-2
```
采用Cmake安装
```bash
mkdir -p build
cd build
cmake -DFIND_CUDA=OFF -DUSE_CHEMIQ=OFF -DUSE_PYQPANDA=OFF ..
make
```
安装后在`/usr/local` 目录下
### 运行
1. g++ 编译
```
g++ main.cpp -std=c++14 -fopenmp -I{QPanda安装路径}/include/qpanda2/ -I{QPanda安装路径}/include/qpanda2/ThirdParty/ -L{QPanda安装路径}/lib/ -lQPanda2 -lComponents -lantlr4 -o out
./out
```
{QPanda安装路径} 可以替换为`/usr/local`
或者提供的下面的脚本
```bash
bash run.sh main.cpp
```

2. cmake 编译 [推荐使用，不容易报错]
在这个目录下，运行
```bash
bash ./build.sh main.cpp
```
即可编译运行。

## 实现Block encodings
在python/blockencoding/ftable.py 下实现了Block encodings，需要转为C++版本。
参考： https://github.com/QuantumComputingLab/fable
## 实现AQC算法
目前在python/AQC.ipynb 中实现了门模型的AQC算法，可以直接运行。

## 文档撰写
在docs/ 目录下撰写markdown文档，后面可以整理成latex文档。