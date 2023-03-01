#!/usr/bin/env bash
#SBATCH --array=0-80%40
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16GB
#SBATCH --time=0:25:00
#SBATCH --cpus-per-gpu=2
#SBATCH --output=sbatch_out/exp_vgg_cifar_noisy_grad.%A.%a.out
#SBATCH --job-name=exp_vgg_cifar_noisy_grad

. /etc/profile
module load anaconda/3
conda activate ffcv

# bias_arr=(-1.00e-02 -2.03e-02 -4.12e-02 -8.38e-02 -1.70e-01 -3.46e-01 -7.02e-01 -1.43e+00 -2.89e+00 -5.88e+00 -1.19e+01 -2.42e+01 -4.92e+01 -1.00e+02)
# bias_arr=(0.0 1.00e-03 3.16e-03 1.00e-02 3.16e-02 1.00e-01 3.16e-01 1.00 3.16 1.00e+01 3.16e+01 1.00e+02)
bias_arr=(0.0 1e-8 1e-7 1e-6 5e-6 1e-5 1e-4 1e-3 1e-2)
# bias_arr=(0.0)
# variance_arr=(0.0 1.00e-02 5.88e-02 3.46e-01 2.03e+00 1.19e+01 7.02e+01 4.12e+02 2.42e+03 1.43e+04 8.38e+04 4.92e+05 2.89e+06 1.70e+07 1.00e+08)
# variance_arr=(1.00e-02 1.00e-01 1.00 1.00e+01 1.00e+02 1.00e+03 1.00e+04 1.00e+05 1.00e+06 1.00e+07 1.00e+08)
variance_arr=(0 1e-3 1e-2 1e-1 1 5 10 40 100)
# variance_arr=(0.0)

lenB=${#bias_arr[@]}
bidx=$((SLURM_ARRAY_TASK_ID%lenB))
vidx=$((SLURM_ARRAY_TASK_ID/lenB))

python train_approx_grad.py --bias=${bias_arr[$bidx]} --variance=${variance_arr[$vidx]} --epochs=50 --logfreq=2
