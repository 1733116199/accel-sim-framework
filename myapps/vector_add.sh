#!/bin/bash
pushd ./$1
rm -rf vector_add &&\
nvcc -o vector_add vector_add.cu &&\
./vector_add
popd