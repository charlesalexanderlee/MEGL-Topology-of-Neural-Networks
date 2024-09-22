![MEGL](https://meglab.wdfiles.com/local--files/home:home/megl_logo_color.png)

# Topology of Neural Networks
GitHub respository for the Topology of Neural Networks team at the Mason Experimental Geometry Laboratory.

https://megl.science.gmu.edu/

## Abstract
A neural network may be geometrically interpreted as nonlinear function that stretches and pulls apart data between vector spaces. If a dataset has interesting geometric or topological structure, one might ask how the structure of the data will change when passed through a neural network. This is achieved by explicitly viewing the dataset as a manifold and observing how the topological complexity (i.e., the sum of the Betti numbers) of the manifold changes as it passes through the activation layers of a neural network. The goal of this project is to study how the topological complexity of the data changes by tuning the hyper-parameters of the network. This enables us to possibly understand the relationship between the structural mechanics of the network and its performance.

## Installation
This repository requires Docker, which can be installed from:

https://docs.docker.com/engine/install/ 

Check that Docker has been successfully installed and that the Docker daemon is currently running:
```
docker --version
docker info
```
If the Docker daemon is running, `docker info` will return detailed information about the Docker system. If it is not running, refer to the Docker daemon documentation:

https://docs.docker.com/engine/daemon/start/

Clone the Github repoistory:
```
git clone https://github.com/charlesalexanderlee/MEGL-Topology-of-Neural-Networks
```
Go into the project folder:
```
cd MEGL-Topology-of-Neural-Networks/
```

Build the Docker container:
```
docker build --platform=linux/amd64 -t megl-tnn .
```

Run the Docker container:
```
docker run -it --rm megl-tnn
```

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
* [Diane Hamilton](https://www.linkedin.com/in/dhamil-bytes/)
* Joseph A. Jung
* [Charles Alexander Lee](https://www.linkedin.com/in/charlesalee/)
* David Wigginton
