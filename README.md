1. Connect to the Informatics VPN.
2. SSH sxxxxxxx@mlp.inf.ed.ac.uk
3. Create the conda environment
 ```
 /opt/conda/bin/conda create -n cell2fire python=3.9
conda activate cell2fire
 ```
4. Clone the Cell2Fire branch of the repository
 ```
 git clone https://github.com/Cell2Fire/Cell2Fire.git
```
5. Setup the Cell2Fire environment on your machine
```
 cd Cell2Fire
 nano ~/.bashrc
```
 and then add the line `export PATH=”/home/{your-student-number}/.local/bin:$PATH”`
Next
```
source ~/.bashrc
pip install -e .
conda install -c conda-forge opencv
pip install ”numpy¡2”
pip install scipy
conda install -c conda-forge gcc make
conda install -c conda-forge eigen
conda install -c conda-forge boost
cd /home/{your-student-number}/Cell2Fire/cell2fire/Cell2FireC/
make
pip install -r requirements.txt
python setup.py develop
```
6. There will be issues with a couple of files, notably with the Stats.py file, and you will have to nano in to the file in order to fix it. Remove all instances of `figsize` in the `plt.savefig()` in the file, and you will have to change the SummaryDF lines.
7. Run the simulator:
```
python main.py –input-instance-folder ../data/Sub40x40/ –output- folder ../results/Sub40x40 –ignitions –sim-years 1 –nsims 5 –finalGrid – weather rows –nweathers 1 –Fire-Period-Length 1.0 –output-messages – ROS-CV 0.0 –seed 123 –stats –allPlots –IgnitionRad 5 –grids –combine
```
 Note, you may have to edit the Makeafile using nano to update the paths for boost, eigen etc.

 8. Finally, setup this repo and submit a batch job:
```
git clone https://github.com/niallmeagher/mlp_cw4
cd mlp_cw4
sbatch batch_job {NumEpochs} {NumEpisodes} {InputFolder} {OutputFolder} --stochastic --normalise --single_sim
```
