#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mail-user=<brdiep@mail.ubc.ca>
#SBATCH --mail-type=ALL

cd ~/$project/literacy_asr

# Set up modules/virtual environment
module purge
module load StdEnv/2023 python/3.11 scipy-stack gcc/12.3 arrow/14 cuda
source ~/std23/bin/activate

python resample_cv.py --code en --set dev --transcript_path ~/scratch/common_voice/transcript/en/dev.tsv --outdir ~/scratch/processed_common_voice/ --dataset_path ~/scratch/common_voice/audio/