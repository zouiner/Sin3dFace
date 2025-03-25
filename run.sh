#!/bin/bash
#SBATCH --job-name=Sin3d_vggface2_8_32_256_32             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=moonlight.dum1@gmail.com          # Where to send mail
#SBATCH --ntasks=3                             # Run a single task...
#SBATCH --cpus-per-task=1                      # ...with a single CPU
#SBATCH --mem=16gb                             # Job memory request
#SBATCH --time=3-00:00:00                      # Time limit (DD-HH:MM:SS)
#SBATCH --output=cuda_log/Sin3d_vggface2_8_32_256_32_%j.log      # Standard output and error log
#SBATCH --error=cuda_error_log/%x-%j.err                # Standard error log
#SBATCH --partition=gpu                        # Select the GPU nodes... (, interactive, gpu , gpuplus)  
#SBATCH --gres=gpu:3                          # ...and the Number of GPUs
#SBATCH --account=its-gpu-2023                 # Run job under project <project>

module purge
module load GCC/12.2.0
module load Miniconda3 # for conda, if using venv you wont need this

# This is just printing stuff you don't really need these lines
echo `date`: executing gpu_test on host $HOSTNAME with $SLURM_CPUS_ON_NODE cpu cores echo 
cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')
echo I can see GPU devices $CUDA_VISIBLE_DEVICES
echo

source ~/.bashrc

conda activate 3dSin # Your conda environment here 
# or source your .venv environment here if using venv

# now run the python script
export OMP_NUM_THREADS=1
torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/Sin3d_vggface2_8_32_256_32.yaml --save_dir logs/Sin3d_vggface2_8_32_256_32 --resume /users/ps1510/scratch/Programs/Sin3dFace/logs/SinSR_vggface2_8_32_256_32/2025-03-16-14-38/ckpts/model_69000.pth
# torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/Sin3d_vggface2_16_64_256_64.yaml --save_dir logs/Sin3d_vggface2_16_64_256_64 --resume /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_16_64_256_64/2025-03-21-02-50/ckpts/model_54000.pth
# torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/Sin3d_vggface2_32_128_256_128.yaml --save_dir logs/Sin3d_vggface2_32_128_256_128 --resume
# torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/Sin3d_vggface2_64_256_256_256.yaml --save_dir logs/Sin3d_vggface2_64_256_256_256 --resume
# python3 -m torch.distributed.run --nproc_per_node=3 main_distill.py --cfg_path configs/Sin3d_vggface2.yaml --save_dir logs/Sin3d

# to run the command: sbatch run.sh