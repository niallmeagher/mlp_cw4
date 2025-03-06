#!/bin/bash

# Define hyperparameters
NumEpochs=1000
NumEpisodes=3
learning_rate=3e-4
clip_epsilon=0.2
gamma=0.99
update_epochs=4
value_loss_coef=0.5
entropy_coef=0.1

# Directory for your Python code
Dir="/disk/scratch/mlp_cw4/"
InputFileDirectory=""
OutputFileDirectory=""

# Run the Python script with all hyperparameters and correct input names
python "$Dir/main.py" --epochs $NumEpochs --episodes $NumEpisodes \
--learning_rate $learning_rate --clip_epsilon $clip_epsilon --gamma $gamma \
--update_epochs $update_epochs --value_loss_coef $value_loss_coef --entropy_coef $entropy_coef \
--input_dir "$InputFileDirectory" --output_dir "$OutputFileDirectory"
