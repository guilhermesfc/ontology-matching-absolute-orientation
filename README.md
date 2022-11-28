# Ontology Matching Through Absolute Orientation of Embedding Spaces

## Installation

It is recommended to use virtual environments to manage your python environment.

You can use e.g. anaconda.

https://www.anaconda.com/products/individual


A recommended setup is to run a Ubuntu inside the WSL2 on Windows 10 as execution engine. You need to ensure Ubuntu has Java installed.
Visual Studio Code can be used as IDE, connected to the WSL2 Ubuntu.

The following bash scripts are to be run on the command line.


First, check out the repository:
```bash
#bash
git clone git@github.com:guilhermesfc/ontology-matching-absolute-orientation.git
```

### Use anaconda to manage your python environment:
```bash
#bash
conda create --name <YOUR_NAME> python=3.8
conda activate <YOUR_NAME>

cd mt-ds-sandbox
pip install -r requirements.txt

cd src
echo `pwd` > ~/anaconda3/envs/<YOUR_NAME>/lib/python3.8/site-packages/mt-sandbox.pth
```

## Configuration
The configurations for the experiments are stored in the ```src/entrypoint/create_exp_series/``` folder:
- Synthetic data: ```config_synth_graph.py```
- OAEI multifarm: ```config_multifarm.py```

You need to edit the _workdir_, _resultdir_ and _java_executable_ paths from these two files.

## Execution

Now you can run the program:
```bash
#bash
cd ..
python src/entrypoint/create_exp_series/create_experiment.py

# Or in VS Code: hit CTRL-F5 on the tab, where create_experiment.py is open.
```
The results of the experiments are available in the _resultdir_ folder.

## Running the jupyter notebook

Create a second virtual environment:

```bash
#bash
conda create --name <YOUR_NAME_2> python=3.8
conda activate <YOUR_NAME_2>

cd mt-ds-sandbox
pip install -r requirements_jupyter.txt

# Now you can start the local jupyter notebook server:
cd notebooks
jupyter notebook

# the notebooks overview page will be opened in your browser 
```

You can adjust and run notebook _Synthetic_data_systematic_checks_ to reproduce the figures mentioned in the paper.

## Running the OAEI multifarm results evaluation

We levereged the [MELT - Matching Evaluation Toolkit](https://github.com/dwslab/melt) for evaluating our approach on the OAEI multifarm dataset. You can replicate this by executing the *MultiRunMain.java* file inside the *absolute-orientation-java-master* folder. The relevant metrics will be shown on the output file *trackPerformanceCube.csv*.

For additional help, please refer to respective MELT [user guide](https://dwslab.github.io/melt/).
