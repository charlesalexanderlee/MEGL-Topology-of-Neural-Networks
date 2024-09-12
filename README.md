![MEGL](https://meglab.wdfiles.com/local--files/home:home/megl_logo_color.png)

# Topology of Neural Networks
GitHub respository for the Topology of Neural Networks team at the Mason Experimental Geometry Laboratory.

https://megl.science.gmu.edu/

## Abstract
A neural network may be geometrically interpreted as nonlinear function that stretches and pulls apart data between vector spaces. If a dataset has interesting geometric or topological structure, one might ask how the structure of the data will change when passed through a neural network. This is achieved by explicitly viewing the dataset as a manifold and observing how the topological complexity (i.e., the sum of the Betti numbers) of the manifold changes as it passes through the activation layers of a neural network. The goal of this project is to study how the topological complexity of the data changes by tuning the hyper-parameters of the network. This enables us to possibly understand the relationship between the structural mechanics of the network and its performance.

## Installation
This repository requires Python 3.10.0. If you do not have this version of Python, you can use `pyenv` to install it:
https://github.com/pyenv/pyenv?tab=readme-ov-file#installation

Clone the Github repoistory:
```
git clone https://github.com/charlesalexanderlee/MEGL-Topology-of-Neural-Networks
```
Go into the project folder:
```
cd MEGL-Topology-of-Neural-Networks/
```
Install Python 3.10.0 with `pyenv` and switch versions:
```
pyenv install 3.10.0
pyenv local 3.10.0
```
After switching to Python 3.10.0, we need to create a virtual environment using `virtualenv`:
```
pip install virtualenv
virtualenv venv
```

MacOS / Linux users:
```
source venv/bin/activate
```

Windows users:
```
source venv/Scripts/activate
```

Finally, we need to install the project dependency requirements:
```
pip install -r freeze_file.txt
```
The installation should be complete. You can now run any experiment in the `src/` directory!

## Running Experiments
Coming soon (we are currently experimenting with the MNIST dataset and the CIFAR10 dataset).

## Authors
**Faculty Member**
* Dr. Benjamin Schweinhart

**Graduate Students**
* Justin Cox
* Shrunal Pothagoni

**Undergraduate Students**
* Eugenie Ahn
* Finn Brennan
* Diane Hamilton
* Joseph A. Jung
* Charles Alexander Lee
* David Wigginton
