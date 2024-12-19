import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import ripserplusplus as rpp
from gudhi.representations import Silhouette
import warnings
import matplotlib.pyplot as plt
import scipy.spatial
import scipy
from scipy import integrate
import wandb
import os

# Define the base directory and sweep number (Change base_dir to the directory you wish to save your sweeps)
sweep_number = 6 # Example sweep number
base_dir = f"/scratch/jjung43/MEGL-Topology-of-Neural-Networks/data4Res/sweep{sweep_number}"

# Specify the epochs and number of layers
epochs = [0, 2, 4, 6, 8, 10]
num_layers = 3  # Assuming 6 layers

# Specify the number of points and dimensions
num_points = 1600               # ** Number of points you wish to sample from the point cloud **
dim = 2                         # compute up to H_{dim} Topological Complexity
# Iterate through each epoch
allTC = [[], [], [], []]        # Add more if you'd like to computer higher dimensional TC, this limits you to 0-3.
Metric2 = []


for epoch in epochs:
    TC = []
    epoch_dir = os.path.join(base_dir, f"epoch{epoch}")
    if not os.path.exists(epoch_dir):
        print(f"Epoch directory {epoch_dir} does not exist. Skipping...")
        continue

    # Load each layer's data
    layer_data = {}
    intermediate_outputs = []
    for layer in range(num_layers):
        layer_file = os.path.join(epoch_dir, f"layer{layer}.npy")

        if os.path.exists(layer_file):
            intermediate_outputs.append(np.load(layer_file))
            layer_data[f"layer{layer}"] = np.load(layer_file)
        else:
            print(f"Layer file {layer_file} is missing. Skipping this layer...")

    outputs = [[], [], [], []]

    for num in range(len(intermediate_outputs)):
        intermediate_outputs[num] = intermediate_outputs[num][:num_points]
        X = intermediate_outputs[num].reshape(intermediate_outputs[num].shape[0], -1) # The Point cloud
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diagram = rpp.run(f"--format point-cloud --dim {dim} --threshold 1000", X)

        for homology in range(len(diagram)):
            if len(diagram[homology]) == 0:
                print(f"Homology {homology}: Empty diagram. Skipping...")
                continue

            # Debugging the diagram structure
            diagram[homology] = np.array([(pt['birth'], pt['death']) for pt in diagram[homology]])
            print(f"Homology {homology}: Converted to NumPy array: {diagram[homology]}")

            # Replace infinite death times
            threshold = -1
            diagram[homology][np.isinf(diagram[homology][:, 1]), 1] = threshold

            diags = [diagram[homology]]
            if len(diags[0]) == 0:
                print(f"Homology {homology}: No valid points for Silhouette. Skipping...")
                continue

            print(f"Prepared diagram for Homology {homology}: {diags}")
            SH = Silhouette(resolution=1000, weight=lambda x: np.power(x[1] - x[0], 1))
            sh = SH.fit_transform(diags)  # Pass the list of diagrams

            outputs[homology].append(np.linalg.norm(sh))

    print(outputs)

    for homology in range(len(outputs)):
        if len(outputs[homology]) > 0:
            tcx = np.linspace(0, 1, len(outputs[homology]))
            allTC[homology].append(np.sqrt(scipy.integrate.simpson([y**2 for y in outputs[homology]], tcx)))


print(allTC)
# Plot and save
plt.title("Topological Complexity Over Training")
plt.xlabel("Epochs")
for homology in range(len(allTC)):
    if allTC[homology]:  # Ensure data is not empty
        print("we got a non-empty TC")
        plt.plot(allTC[homology], label=f"Norm of TC_{homology}")
    else:
        print(f"No data for Homology {homology}")

# Define custom x-axis ticks and labels
custom_ticks = np.linspace(1, 5, len(epochs))  # Positions for labels on the x-axis
custom_labels = [0, 2, 4, 6, 8, 10]         # Labels corresponding to epochs
plt.xticks(custom_ticks, labels=custom_labels)
# Label the axes
plt.xlabel("Epochs")
plt.legend()  # Add a legend to identify the lines
plt.savefig(f"Resnet-sweep{sweep_number}-{num_points}pts-dim2")  # Save the plot
plt.show()  # Display the plot
