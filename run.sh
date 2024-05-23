#!/bin/bash
g++ $1 -std=c++14 -fopenmp -I/usr/local/include/qpanda2/ -I/usr/local/include/qpanda2/ThirdParty/ -L/usr/local/lib/ -lQPanda2 -lComponents -lantlr4 -o out
./out $2