#!/bin/bash
# Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# sbatch batchjob.sh 1000 10 Sub20x20 Sub20x20 --stochastic --normalise --single_sim
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b slurm_arrayjob.sh -e experiments.txt -m 12
# ```

# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
#SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=14000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=2

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=08:00:00


# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
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

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=cell2fire
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}


# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
data_path=/home/${USER}



# input data directory path on the scratch disk of the node
# dest_path=${SCRATCH_HOME}/${input_folder}
# mkdir -p ${dest_path}  # make it if required

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html

# send over both mlp_cw4 and Cell2Fire folders

rsync --archive --update --compress --progress ${data_path}/mlp_cw4 ${SCRATCH_HOME}
rsync --archive --update --compress --progress ${data_path}/Cell2Fire ${SCRATCH_HOME}


# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

NumEpochs=$1 # eg 1000
NumEpisodes=$2 # eg 10
InputFolder=$3
OutputFolder=$4
InputFileDirectory=${SCRATCH_HOME}/Cell2Fire/data/${InputFolder} # eg Sub20x20
OutputFileDirectory=${SCRATCH_HOME}/Cell2Fire/results/${OutputFolder} # eg Sub20x20

# Process additional flags
shift 4
stochastic_flag=""
normalise_flag=""
single_sim_flag=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stochastic)
            stochastic_flag="--stochastic"
            shift
            ;;
        --normalise)
            normalise_flag="--normalise"
            shift
            ;;
        --single_sim)
            single_sim_flag="--single_sim"
            shift
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
done

COMMAND="python ${SCRATCH_HOME}/mlp_cw4/main.py -n ${NumEpochs} -e ${NumEpisodes} -i "${InputFileDirectory}" -o "${OutputFileDirectory}" ${stochastic_flag} ${normalise_flag} ${single_sim_flag}"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"


Output=${SCRATCH_HOME}/Cell2Fire/data/${InputFolder}_Test/Checkpoints
results=${OutputFileDirectory}
src_path=${Output}
dest_path=${data_path}/mlp_cw4/results/${OutputFolder}

mkdir -p ${dest_path}
rsync  --archive --update --compress --progress ${src_path}/ ${dest_path}/
rsync  --archive --update --compress --progress ${results}/ ${dest_path}/

# Delete folders from scratch space
rm -rf ${SCRATCH_HOME}

# =========================
# Post experiment logging
# =========================
echo ""
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
