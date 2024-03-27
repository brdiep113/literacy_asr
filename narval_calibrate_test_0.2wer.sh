#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mail-user=<brdiep@mail.ubc.ca>
#SBATCH --mail-type=ALL

cd ~/$project/literacy_asr

# Set up modules/virtual environment
module purge
module load StdEnv/2023 python/3.11 scipy-stack gcc/12.3 arrow/14 cuda
source ~/std23/bin/activate

python narval_calibrate_test.py --beams 10 --sentences 10 --wer 0.2 --alpha 0.2