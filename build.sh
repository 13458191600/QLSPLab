#!/bin/bash
export SCRIPFILE=$1
echo "Compiling $SCRIPFILE"
mkdir -p build
cd build
cmake -D SCRIPFILE=$1 ..
make -j4
cd bin
./testQPanda
echo "DONE"