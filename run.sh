#!/bin/bash
#SBATCH --job-name=SinSR_vggface2_64_256_256_256             # Job name
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=moonlight.dum1@gmail.com          # Where to send mail
#SBATCH --ntasks=3                             # Run a single task...
#SBATCH --cpus-per-task=1                      # ...with a single CPU
#SBATCH --mem=16gb                             # Job memory request
#SBATCH --time=3-00:00:00                      # Time limit (DD-HH:MM:SS)
#SBATCH --output=cuda_log/SinSR_vggface2_64_256_256_256_%j.log      # Standard output and error log
#SBATCH --error=cuda_error_log/SinSR_64_256_256_256_%x-%j.err               # Standard error log
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
# torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/SinSR_vggface2_8_32_256_32.yaml --save_dir logs/SinSR_vggface2_8_32_256_32
# torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/SinSR_vggface2_16_64_256_64.yaml --save_dir logs/SinSR_vggface2_16_64_256_64
# torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/SinSR_vggface2_32_128_256_128.yaml --save_dir logs/SinSR_vggface2_32_128_256_128
torchrun --nproc_per_node=3 main_distill.py --cfg_path configs/SinSR_vggface2_64_256_256_256.yaml --save_dir logs/SinSR_vggface2_64_256_256_256
# python3 -m torch.distributed.run --nproc_per_node=3 main_distill.py --cfg_path configs/SinSR_vggface2.yaml --save_dir logs/SinSR

# to run the command: sbatch run.sh