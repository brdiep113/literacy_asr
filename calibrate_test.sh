#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100L:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mail-user=<brdiep@mail.ubc.ca>
#SBATCH --mail-type=ALL

cd ~/$project/literacy_asr

# Set up modules/virtual environment
module purge
module load StdEnv/2023 python/3.11 scipy-stack gcc/12.3 arrow/14 cuda
source ~/std23/bin/activate

# Login to W&B to log metrics
wandb login $API_KEY

python calibrate_test.py --projectname test_calibration --token $HUGGINGFACE_TOKEN --beams 10 --sentences 10