#!/bin/bash
#SBATCH --job-name=zzzz
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --mem=100000  
#SBATCH -N 1


python main_bronze.py 

# 取消当前作业以释放节点
scancel $SLURM_JOB_ID