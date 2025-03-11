#!/bin/bash

# Define variables
NumEpochs=1000
NumEpisodes=3
Dir="/disk/scratch//mlp_cw4/"
InputFileDirectory=""
OutputFileDirectory=""

# Call the Python script with arguments
python "$Dir/main.py" --epochs $NumEpochs --episodes $NumEpisodes --input_dir "$InputFileDirectory" --output_dir "$OutputFileDirectory"