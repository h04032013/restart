#!/bin/bash
#SBATCH --job-name=baselin
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --mem=128G
#SBATCH --output=genb_output.out
#SBATCH --error=genb_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu


cd /n/netscratch/dam_lab/Lab/hdiaz/ft_project


module purge
module load Mambaforge
module load cuda cudnn
mamba activate env11

# Run training
python main.py
