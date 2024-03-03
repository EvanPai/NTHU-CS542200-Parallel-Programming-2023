#!/bin/bash
#SBATCH --job-name=kmeans_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --gres=gpu:1

#SBATCH --output=output_test.log
#SBATCH --error=error_test.log

# Run KMeans sequential
echo "Running KMeans..."
time srun -p prof -N1 -n1 ./KMeans

# Run KMeans CUDA 
echo "Running KMeans_CUDA..."
time srun -p prof -N1 -n1 --gres=gpu:1 ./KMeans_CUDA

# Run KMeans OpenMP 
echo "Running KMeans_OpenMP..."
time srun -p prof -N1 -n1 -c2 ./KMeans_OpenMP

#echo "Profiling KMeans_CUDA..."
#time srun -p prof -N1 -n1 --gres=gpu:1 nvprof ./KMeans_CUDA



# Run KMeans hybrid 
# echo "Running KMeans_hybrid..."
# time srun -p prof -N1 -n1 -c4 --gres=gpu:1 ./KMeans_hybrid

# Run KMeans with OpenACC
##echo "Running KMeans with OpenACC..."
##time srun -N1 -n1 --gres=gpu:1 ./KMeans_OpenACC

# 在hades02上跑
# time ./KMeans

# 目前KMeans code是錯的