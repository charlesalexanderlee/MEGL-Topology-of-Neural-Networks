import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from ripser import ripser
from gudhi.representations import Silhouette
import warnings
import matplotlib.pyplot as plt
import scipy.spatial
import wandb

class ExtractIntermediateOutputs(Callback):
    def __init__(self, model, epochs_to_extract, layer_names, x_train, save_location):
        super(ExtractIntermediateOutputs, self).__init__()
        self.model = model
        self.epochs_to_extract = epochs_to_extract
        self.layer_names = layer_names
        self.x_train = x_train
        self.save_location = save_location
        wandb.init()
        wandb.define_metric("tc", summary='min')
        wandb.define_metric("Metric2", summary='min')

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs_to_extract:
            intermediate_model = []
            intermediate_outputs = []
            for layer in range(len(self.layer_names)):
                intermediate_model.append(tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.layer_names[layer]).output))
                intermediate_outputs.append(intermediate_model[layer].predict(self.x_train))
            
            output = []
            
            for num in range(len(intermediate_outputs)):
                intermediate_outputs[num] = intermediate_outputs[num][:100]
                maxdim = 3
                threshold = 1000 
                normalization='identity'
                metric='L2'
                k=12
                percentile=0.9
                center=None
                _distance = lambda u, v: np.sqrt(((u-v)**2).sum())
                X = intermediate_outputs[num].reshape(intermediate_outputs[num].shape[0], -1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric=_distance))
                    diagram = ripser(distance_matrix, maxdim=maxdim, thresh=threshold, metric='precomputed')['dgms']
                if diagram[0][-1][-1] == np.inf:
                    diagram[0][-1][-1] = threshold
                dx=0.1
                min_x= 0
                max_x=10
                threshold=-1
                diags = [diagram[0][:-1]]
                SH = Silhouette(resolution=1000, weight=lambda x: np.power(x[1]-x[0],1))
                sh = SH.fit_transform(diags)

                output.append(np.linalg.norm(sh))
            
            tcx = np.linspace(0, 1, len(output))

            """
            plt.plot(outx, output)
            plt.title("Silhouette")
            plt.savefig(self.save_location + "_fig_epoch" + str(epoch) + ".png")"""

            tc = np.sqrt(scipy.integrate.simpson([y**2 for y in output], tcx))

            met2 = np.sqrt(scipy.integrate.simpson([y**2 for y in output[-2:]], tcx[-2:]))/np.sqrt(scipy.integrate.simpson([y**2 for y in output[:-2]], tcx[:-2]))
            log_dict = {
                "tc": tc,
                "Metric2" : met2
            }
            wandb.log(log_dict)
