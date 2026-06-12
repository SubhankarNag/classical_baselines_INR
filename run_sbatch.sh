#!/bin/bash -x
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=18
#SBATCH --nodelist=cn39-a40
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH -J CIMGC
#SBATCH -t 96:00:00
#SBATCH -o logs/%j.out               # name of stdout output file(--output)
#SBATCH -e logs/%j.err               # name of stderr error file(--error)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=24d0373@iitb.ac.in


conda activate mri_codec


cd $SLURM_WORKDIR
cd codes/mri_compression_all/classical_baselines_INR/ || exit



# --- TIMER START ---
START_TIME=$(date +%s)
echo "Starting Execution at: $(date)"



CUDA_VISIBLE_DEVICES=0 python -u run_rd_curves_V2.py --files fs_0045_3T.h5 fs_0095_1_5T.h5 fs_0069_1_5T.h5 fs_0057_1_5T.h5 fs_0056_1_5T.h5 fs_0060_1_5T.h5 fs_0068_1_5T.h5 fs_0074_1_5T.h5 fs_0053_1_5T.h5 fs_0063_1_5T.h5 fs_0012_3T.h5



# Capture exit code immediately
EXIT_CODE=$?

# --- TIMER END ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate hours, minutes, seconds
HOURS=$((DURATION / 3600))
MINS=$(( (DURATION % 3600) / 60 ))
SECS=$((DURATION % 60))

echo "================================================="
echo "Job finished with exit code $EXIT_CODE"
echo "Finished at: $(date)"
echo "Total Runtime: ${HOURS}h ${MINS}m ${SECS}s"
echo "================================================="