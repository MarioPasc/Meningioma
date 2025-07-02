#!/usr/bin/env bash
#
#SBATCH --job-name=UniRes-<DATASET>      # e.g. UniRes-3mm
#SBATCH --time=03:00:00                  # wall clock limit
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
#SBATCH --error=log-unires.%J.err
#SBATCH --output=log-unires.%J.out

# -------- USER-ADJUSTABLE ONE-LINER ------------------------------------ #
DATASET="3mm"             # ← change to 5mm / 7mm in the other two scripts
# ----------------------------------------------------------------------- #

module load miniconda
source activate unires            

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NITORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK}

INPUT_ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/meningiomas/${DATASET}"
OUTPUT_ROOT="/mnt/home/users/tic_163_uma/mpascual/execs/UNIRES/${DATASET}"

echo "---------- JOB SUMMARY ----------------------------------------"
echo "Host   : $(hostname)"
echo "GPU    : ${CUDA_VISIBLE_DEVICES}"
echo "Input  : ${INPUT_ROOT}"
echo "Output : ${OUTPUT_ROOT}"
echo "--------------------------------------------------------------"
echo

python run_unires_batch.py \
       --input-dir  "${INPUT_ROOT}" \
       --output-dir "${OUTPUT_ROOT}" \
       --device     cuda \
       --threads    "${SLURM_CPUS_PER_TASK}"
