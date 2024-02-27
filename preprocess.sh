#!/bin/bash
#SBATCH --mem=64G
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:0:0    
#SBATCH --mail-user=<brdiep@mail.ubc.ca>
#SBATCH --mail-type=ALL

cd ~/projects/def-lingjzhu/brdiep

# Set up modules/virtual environment
module purge
module load StdEnv/2023 python/3.11 scipy-stack gcc/12.3 arrow/14
source ~/preprocess/bin/activate

python preprocess_common_voice_data.py