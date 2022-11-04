# PR-DNS_base

#Install on lambda2
./clean.sh
export PATH=/work/atif/packages/petsc-3.16.0/lib/petsc/bin/:$PATH
cmake -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .
