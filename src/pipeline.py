# Still far from finished or clean, still having issues implementing the R library

import warnings
import os
import matplotlib.pyplot as plt
#from gph.python import ripser_parallel
from ripser import ripser
import gudhi
from gudhi.representations import Silhouette
#import rpy2
#from rpy2.robjects.packages import importr
import numpy as np

#import visualize
# some useful computational geometry tools
#from sklearn.neighbors import NearestNeighbors
import scipy.spatial
#import resource
import distances

#compute_landscape.tdatools = importr('tdatools')
data = []
for num in range(6):
    data.append(np.load(os.getcwd() + "\data\mnist_long_model_layer_f" + str(num) + "_relu_epoch0.npy")[:100])

output = []

print(data[0])

for num in range(len(data)):
    maxdim = 2 # we decided to make the maxdim 2
    threshold = 1000 
    normalization='identity'
    metric='L2'
    k=12
    percentile=0.9
    center=None
    _distance = lambda u, v: np.sqrt(((u-v)**2).sum())
    X = data[num].reshape(data[num].shape[0], -1)
    with warnings.catch_warnings():
        # supress ripser warnings
        warnings.simplefilter("ignore")
        distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric=_distance))
        diagram = ripser(distance_matrix, maxdim=maxdim, thresh=threshold, metric='precomputed')['dgms']

    print("Diagram " + str(num) + " Done.")

    #print(diagram)

    if diagram[0][-1][-1] == np.inf:
        diagram[0][-1][-1] = threshold

    #print(diagram)

    dx=0.1
    min_x= 0
    max_x=10
    threshold=-1

    diags = [diagram[0][:-1]]

    SH = Silhouette(resolution=1000, weight=lambda x: np.power(x[1]-x[0],1))
    sh = SH.fit_transform(diags)

    #print(sh)

    output.append(np.linalg.norm(sh))

    print("Landscape " + str(num) + " Done.")


# please rewrite this 
outx = []
for i in range(len(output)):    
    outx.append(i / (len(output) - 1))

plt.plot(outx, output)
plt.title("Silhouette")

plt.show()