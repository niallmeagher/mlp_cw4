#!/bin/bash
#SBATCH --job-name=optuna_20x20
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out
#SBATCH --array=1-100%4     # 100 trials total, 4 concurrent
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1         # 1 GPU per trial
#SBATCH --mem=14000
#SBATCH --cpus-per-task=2

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Shared SQLite path (MUST be on network storage!)
DB_PATH="/shared_storage/optuna.db"

# Add random delay (0-5s) to prevent SQLite lock collisions
sleep $(( RANDOM % 5 ))

# Run with retry logic for SQLite busy errors
max_retries=3
retry_count=0

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
# Shared SQLite path (MUST be on network storage!)
DB_PATH="/home/${USER}/shared_storage/optuna.db"

while [ $retry_count -lt $max_retries ]; do
    python mlp_cw4/hyp_tuning.py --db-path "$DB_PATH"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        break
    elif [ $exit_code -eq 5 ]; then  # SQLite locked error
        sleep $(( (RANDOM % 10) + 1 ))
        ((retry_count++))
    else
        exit $exit_code
    fi
done

if [ $retry_count -eq $max_retries ]; then
    echo "Max retries reached for SQLite lock"
    exit 1
fi

# Move output data from scratch to DFS
echo "Moving output data back to DFS"

src_path=${SCRATCH_HOME}/${USER}/Cell2Fire/results/Sub20x20/
dest_path=${data_path}/mlp_cw4/results/
rsync  --archive --update --compress --progress ${src_path}/ ${dest_path}

# Delete folders from scratch space
rm -rf ${SCRATCH_HOME}
