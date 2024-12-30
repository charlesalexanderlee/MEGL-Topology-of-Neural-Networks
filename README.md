![MEGL](https://meglab.wdfiles.com/local--files/home:home/megl_logo_color.png)

# Topology of Neural Networks
GitHub respository for the Topology of Neural Networks team at the Mason Experimental Geometry Laboratory.

https://megl.science.gmu.edu/

## Abstract
A neural network may be geometrically interpreted as nonlinear function that stretches and pulls apart data between vector spaces. If a dataset has interesting geometric or topological structure, one might ask how the structure of the data will change when passed through a neural network. This is achieved by explicitly viewing the dataset as a manifold and observing how the topological complexity of the data changes as it passes through the layers of a neural network and at different epochs of training. We can model the data as a manifold using [persistent homology](https://en.wikipedia.org/wiki/Persistent_homology) which roughly captures the topological structure of the manifold our data lives on. The goal of this project is to study how the topological complexity of our data changes as it goes through each layer of the neural network and as it trains over a certain number of epochs. This enables us to possibly understand the relationship between the structural mechanics of the network and its performance.

## Experiment
Our experiment was inspired by [*Activation Landscapes as a Topological Summary of Neural Network Performance*](https://doi.org/10.48550/arXiv.2110.10136). We can see that in their experiment, the average norm of the activation landscapes, which describe the topological complexity, tend to increase as model accuracy increases. We wanted to run experiments to see if this increase in topological complexity was driven by the 0-th dimensional homology group, or if there was some higher dimension that was causing this increase in complexity. We have shared how to run our experiment below but you can also see our results in the *Experiment_Graphs* directory where each graph shows that the increase in topological complexity is driven by the 0th dimension. 

## Installation

First Ensure that you have python 3.10.x installed.

Clone the Github repository:
```bash
git clone https://github.com/charlesalexanderlee/MEGL-Topology-of-Neural-Networks
```
Go into the project folder:
``` bash
cd MEGL-Topology-of-Neural-Networks/
```

Set up a Python Virtual Environment
``` bash
python -m venv /path/to/new/virtual/environment
```
Activate Environment
``` bash
source /path/to/new/virtual/environment/bin/activate
```
Ensure pip is up to date
``` bash
pip install --upgrade pip
```
Install Dependencies
``` bash
pip install requirements.txt
```
**NOTE: If you have trouble installing Ripser++ refer to their documentation here: 
https://github.com/simonzhang00/ripser-plusplus

## Usage
* Modify the *constants.py* file by changing the relevant variable values. 
If you plan on running the code on a cluster, modify the megl-run.slurm file to fit your specifications.
Otherwise, you can run the following either the SimpleCNN.py or ResNetModel.py files to to train the model and save the data to disk. Where this data will be saved to will depend on the *directory_name* variable in *constants.py*
example command:
``` bash
python SimpleCNN
```

After saving your data to disk, run the respective calculation file (Either RipserCalculations.py or ResNetRipserCalculations.py). We have also provided a slurm file to use if running on a compute cluster. 
example command:
``` bash
python RipserCalculations.py
```
This will save a graph similar to what you see in *Experiment_Graphs*.


## Authors
**Faculty Member**
* Dr. Benjamin Schweinhart

**Graduate Students**
* Shrunal Pothagoni

**Undergraduate Students**
* Eugenie Ahn
* Finn Brennan
* [Diane Hamilton](https://www.linkedin.com/in/dhamil-bytes/)
* Joseph A. Jung
* [Charles Alexander Lee](https://www.linkedin.com/in/charlesalee/)
* David Wigginton
