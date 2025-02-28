#!/bin/bash

# Define variables
NumEpochs=1000
NumEpisodes=3
Dir="/home/s2750265/mlp_cw4/"
InputFileDirectory=""
OutputFileDirectory=""

# Call the Python script with arguments
python "$Dir/main.py" --epochs $NumEpochs --episodes $NumEpisodes --input_dir "$InputFileDirectory" --output_dir "$OutputFileDirectory"

