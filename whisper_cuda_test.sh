#!/bin/bash
#SBATCH --mem=64G
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:0:0    
#SBATCH --mail-user=<brdiep@mail.ubc.ca>
#SBATCH --mail-type=ALL

cd ~/$project/whisper-test

# Set up modules/virtual environment
module purge
module load StdEnv/2020 python/3.11 scipy-stack gcc/9.3 arrow cuda
source ~/cuda-test/bin/activate

# Login to W&B to log metrics
wandb login $API_KEY

python whisper_tutorial.py --projectname test_cuda --language hi --token $HUGGINGFACE_TOKEN