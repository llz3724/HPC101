#!/bin/bash
#SBATCH --job-name=solver
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=52
#SBATCH --time=00:00:10
#SBATCH --partition=solver2

# 打印作业信息，方便调试
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "----------------------------------------"

# 先加载 spack 再加载编译环境
source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi

export I_MPI_PMI_LIBRARY=/slurm/libpmi2.so.0.0.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export I_MPI_PIN=1
export I_MPI_FABRICS=shm:ofi
# 3. 设置每个 MPI 进程的绑核域为 OMP_NUM_THREADS 的大小
export I_MPI_PIN_DOMAIN=omp

# 4. 开启绑核调试信息，方便我们检查绑核结果
export I_MPI_DEBUG=5
# mpirun ./build/bicgstab $1
srun --mpi=pmi2 --ntasks-per-node=1 --cpus-per-task=52 ./build/bicgstab ./data/case_2001.bin $1

echo "Job finished."