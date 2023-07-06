# PR-DNS_base
# Author(s) : Atif

On BNL IC, inside PR_DNS_Base/DNS
conda activate pr-dns
module load cmake/3.16.2 gcc/9.3.0 openmpi/4.1.1-gcc-9.3.0 cuda/11.1 petsc/3.16.0-gcc-9.3.0 python/3.9-anaconda-2022-5
export PATH=/hpcgpfs01/software/petsc/3.16.0/lib/petsc/bin/:$PATH
./clean.sh
cmake -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc .
make -j8
cd climate

salloc -p debug -A dns -N 1 -t 00:30:00
conda activate pr-dns
mpirun --mca btl_openib_allow_ib 1 -n 16 ./climate -d 3 -p 4 2 2 -i input-pr-dns/in-hackathon-meifeng -o output-hackathon

export TAU_COMM_MATRIX=1
export TAU_TRACE=1
export TAU_CALLPATH_DEPTH=4

For libfabric.so.1
export LD_LIBRARY_PATH=/hpcgpfs01/software/spack2/spack/opt/spack/linux-rhel7-broadwell/gcc-9.3.0/libfabric-1.13.2-ggbtb5xqcodbkzpdwl2uzqfewqrbvxad/lib:$LD_LIBRARY_PATH

mpirun --mca btl_openib_allow_ib 1 -n 16 tau_exec -T mpi,cupti -ebs -cupti /work/atif/PR_DNS_base/DNS/climate/climate -d 3 -p 4 2 2 -i input-pr-dns/in-entrainment3dd_short -o out-pr-dns

#Install on lambda2
conda activate pr-dns
./clean.sh
export PATH=/work/atif/packages/petsc-3.16.0/lib/petsc/bin/:$PATH
cmake -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .

#Install on lambda4
export PATH=/work/atif/petsc-3.16.0/lib/petsc/bin/:$PATH
cmake -DCMAKE_C_COMPILER=/work/atif/packages/openmpi-4.0.3-lambda4/bin/mpicc -DCMAKE_CXX_COMPILER=/work/atif/packages/openmpi-4.0.3-lambda4/bin/mpicxx -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .

/work/atif/packages/openmpi-4.0.3-lambda4/bin/mpirun --mca btl_openib_allow_ib 1 -n 1 ./climate/climate -d 3 -p 1 1 1 -i ./climate/input-pr-dns/in-entrainment3dd_case1_vlm_test1 -o output-cuda -mat_type aijcusparse -vec_type cuda -log_view

