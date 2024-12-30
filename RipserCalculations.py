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
from constants import directory_name

# ======= DEFINE CALCULATION PARAMETERS =======
num_points = 400
sweep_number = 0  # Example sweep number
base_dir = f"~/MEGL-Topology-of-Neural-Networks/{directory_name}/sweep{sweep_number}"      
num_layers = 6                                    # Careful with changing this, You must make sure your model actually has this number of dense layers
num_dimensions = 2
# =============================================

# Define the base directory and sweep number


# Specify the epochs and number of layers
epochs = [0, 9, 19, 29, 39, 49]

allTC = [[], [], [], []]
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
            diagram = rpp.run(f"--format point-cloud --dim {num_dimensions} --threshold 1000", X)

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

            SH = Silhouette(resolution=1000, weight=lambda x: np.power(x[1] - x[0], 1))
            sh = SH.fit_transform(diags)  # Pass the list of diagrams

            outputs[homology].append(np.linalg.norm(sh))

    for homology in range(len(outputs)):
        if len(outputs[homology]) > 0:
            tcx = np.linspace(0, 1, len(outputs[homology]))
            allTC[homology].append(np.sqrt(scipy.integrate.simpson([y**2 for y in outputs[homology]], tcx)))
# ======================== AFTER THIS POINT WE HAVE FINISHED RIPSER++ CALCULATIONS OF PERSISTENT HOMOLOGY =====================

# ====================== Graph Data with Matplotlib ========================

plt.title("Topological Complexity Over Training")
plt.xlabel("Epochs")
for homology in range(len(allTC)):
    if allTC[homology]:  # Ensure data is not empty
        plt.plot(allTC[homology], label=f"Norm of TC_{homology}")
    else:
        print(f"No data for Homology {homology}")

# Define custom x-axis ticks and labels
custom_ticks = np.linspace(1, 5, len(epochs))  # Positions for labels on the x-axis
custom_labels = [0, 9, 19, 29, 39, 49]         # Labels corresponding to epochs
plt.xticks(custom_ticks, labels=custom_labels)
# Label the axes
plt.xlabel("Epochs")
plt.legend()  # Add a legend to identify the lines
plt.savefig(f"Experiment_Plots/sweep{sweep_number}-{num_points}pts-dim3")  # Save the plot
print("Plot saved as 'sweepImage.png'")
plt.show()  # Display the plot

# ===========================================================================
