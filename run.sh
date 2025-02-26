if [ -z "$1" ]; then
  echo "Usage: $0 <input_instance_folder>"
  exit 1
fi

INPUT_DIR="$1"

python main.py \
  --input-instance-folder "$INPUT_DIR" \
  --outputfolder ../results/Sub40x40 \
  --ignitions \
  --sim-years 1 \
  --nsims 5 \
  --finalGrid \
  --weather-rows \
  --nweathers 1 \
  --Fire-Period-Length 1.0 \
  --output-messages \
  --ROS-CV 0.0 \
  --seed 123 \
  --stats \
  --allPlots \
  --IgnitionRad 5 \
  --grids \
  --combine

