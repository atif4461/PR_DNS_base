#PBS -q batch
#PBS -m abe
#PBS -l nodes=2:ppn=32
#PBS -l walltime=960:00:00
#PBS -j oe
#PBS -N dns_3d
#PBS -o /home/tzhang/tmp/$PBS_JOBID.out
##PBS -M tzhang@bnl.gov
#PBS -n

ulimit -s unlimited
#export OMP_STACKSIZE=2000M
#export MP_STACK_SIZE=2000M
#export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH="/home/tzhang/soft/jpeg-9c/lib/":$LD_LIBRARY_PATH

cd /home/tzhang/PR_DNS_base/DNS/climate/
rm -rf out-entrainment3dd_case1
/home/tzhang/soft/openmpi-4.1.1/bin/mpirun --mca  btl_openib_allow_ib 1  -np 64 /home/tzhang/PR_DNS_base/DNS/climate/climate -d 3 -p 4 4 4 -i input-pr-dns/in-entrainment3dd_case1 -o out-entrainment3dd_case1
