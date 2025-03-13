#!/bin/bash
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

CONDA_ENV_NAME=cell2fire
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

# Move input data to scratch disk
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
data_path=/home/${USER}
rsync --archive --update --compress --progress --exclude='results/' ${data_path}/mlp_cw4 ${SCRATCH_HOME}
rsync --archive --update --compress --progress ${data_path}/Cell2Fire ${SCRATCH_HOME}

# Run command
COMMAND="python ${SCRATCH_HOME}/mlp_cw4/hyp_tuning.py --storage "sqlite:///optuna.db""
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

# Move output data from scratch to DFS
echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/${USER}/Cell2Fire/results/Sub20x20/
dest_path=${data_path}/mlp_cw4/results/
rsync  --archive --update --compress --progress ${src_path}/ ${dest_path}

# Delete folders from scratch space
rm -rf ${SCRATCH_HOME}
