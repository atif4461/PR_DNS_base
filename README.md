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





#### alpha1
conda activate pr-dns
./clean.sh
export PATH=/work/atif/petsc-3.16.0/lib/petsc/bin/:$PATH
/work/atif/packages/cmake-3.25.0-linux-x86_64/bin/cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .
make -j16
mpirun --mca btl_openib_allow_ib 1 -n 8 ./climate/climate -d 3 -p 2 2 2 -i ./climate/input-pr-dns/in-entrainment3dd_case1_vlm_test1 -o output-thermod-8 >& thermod.log &

#### Install on lambda2
conda activate pr-dns
./clean.sh
export PATH=/work/atif/packages/petsc-3.16.0/lib/petsc/bin/:$PATH
cmake -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_C_COMPILER=mpicc -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .

#### Install on lambda4
export PATH=/work/atif/petsc-3.16.0/lib/petsc/bin/:$PATH
cmake -DCMAKE_C_COMPILER=/work/atif/packages/openmpi-4.0.3-lambda4/bin/mpicc -DCMAKE_CXX_COMPILER=/work/atif/packages/openmpi-4.0.3-lambda4/bin/mpicxx -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc .

###############
## CORI
###############
module load python3/3.9-anaconda-2021.11
conda activate pr-dns
export PATH=/global/homes/a/atif/packages/openmpi-4.1.1/bin/:$PATH
export PATH=/global/homes/a/atif/packages/petsc-3.16.0/lib/petsc/bin/:$PATH
cmake -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc .
srun /global/homes/a/atif/packages/openmpi-4.1.1/bin/mpirun --mca btl_openib_allow_ib 1 -n 8 ./climate/climate -d 3 -p 2 2 2 -i climate/input-pr-dns/in-entrainment3dd_case1 -o climate/out-pr-dns/
/work/atif/packages/openmpi-4.0.3-lambda4/bin/mpirun --mca btl_openib_allow_ib 1 -n 1 ./climate/climate -d 3 -p 1 1 1 -i ./climate/input-pr-dns/in-entrainment3dd_case1_vlm_test1 -o output-cuda -mat_type aijcusparse -vec_type cuda -log_view

###############
## PERLMUTTER 
###############
Currently Loaded Modules:
  1) craype-x86-milan     4) xpmem/2.6.2-2.5_2.38__gd067c3f.shasta   7) PrgEnv-gnu/8.3.3      (cpe)   10) cray-mpich/8.1.25 (mpi)  13) perftools-base/23.03.0 (dev)
  2) libfabric/1.15.2.0   5) craype-accel-nvidia80                   8) cray-dsmml/0.2.2              11) craype/2.7.20     (c)    14) cudatoolkit/11.7       (g)
  3) craype-network-ofi   6) gpu/1.0                                 9) cray-libsci/23.02.1.1 (math)  12) gcc/11.2.0        (c)    15) cpe/23.03              (cpe)
module load cpe/23.03
conda activate pr-dns
export PATH=/global/homes/a/atif/packages/petsc-3.20.4-cudaaware-64bit/lib/petsc/bin/:$PATH
cmake -DCMAKE_CXX_COMPILER=CC -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_FLAGS="-target-accel=nvidia80" .
salloc --nodes 1 --qos interactive --time 1:00:00 --constraint gpu --gpus 4 --account=m2845
sbatch submit.sh
srun -n 1 ./climate/climate -d 3 -p 1 1 1 -i ./climate/input-pr-dns/in-entrainment3dd_case1_petsc_cuda -o out-petsc-cuda -pc_type bjacobi -mat_type aijcusparse -vec_type cuda -log_view |& tee srun.log &
mpirun --mca opal_common_ucx_opal_mem_hooks 1 --mca btl_openib_allow_ib 1 -n 8 ./climate/climate -d 3 -p 2 2 2 -i ./climate/input-pr-dns/in-entrainment3dd_case1 -o climate/out-pr-dns/ -log_view |& tee srun.log &
