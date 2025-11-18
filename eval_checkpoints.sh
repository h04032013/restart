#!/bin/bash
#SBATCH --job-name=7_50_eval
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=7-0:00:00
#SBATCH --mem=64G
#SBATCH --output=50_output.out
#SBATCH --error=50_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /n/netscratch/dam_lab/Lab/hdiaz/ft_project
module purge
module load Mambaforge
module load cuda cudnn
mamba activate env9

#DIR="${1:-./hgf_new_hub/lr_7e5_mgn_neweval}"
DIR="/n/netscratch/dam_lab/Lab/hdiaz/ft_project/hgf_new_hub/7e5_subset3m"

echo "Evaluating checkpoints in: $DIR"
echo ""

for ckpt in "$DIR"/checkpoint-*; do
    [ -d "$ckpt" ] || continue
    echo "$(basename $ckpt):"
    python simple_eval.py "$ckpt" --adapter --n 4754
    echo ""
done

if [ -d "$DIR/merged_model" ]; then
    echo "merged_model:"
    python simple_eval.py "$DIR/merged_model" --n 4754
    echo ""
fi
