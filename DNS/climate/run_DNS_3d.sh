#!/bin/bash
##SBATCH -p skydebug
##SBATCH -p sky
#SBATCH -p debug
#SBATCH -A dns
#SBATCH -J dns

#SBATCH  --nodes=2
#SBATCH  --output=debug.%j
#SBATCH  --error=error.%j
#SBATCH  --exclusive
##SBATCH  --mail-type=ALL
##SBATCH  --mail-user=tzhang@bnl.gov
#SBATCH  --time=0:10:00


ulimit -s unlimited

cd /sdcc/u/tzhang/dns/DNS_install/cori/DNS/climate
mpirun -n 64 /sdcc/u/tzhang/dns/DNS_install/cori/DNS/climate/climate -d 3 -p 4 4 4 -i input-pi-chamber/in-control-shorter -o out-control-shorter |& tee srun.log
#srun -n 64 ./hello
#cat /proc/cpuinfo
