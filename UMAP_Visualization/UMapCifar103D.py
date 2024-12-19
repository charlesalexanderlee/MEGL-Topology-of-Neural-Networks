import tensorflow as tf
import numpy as np
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_data = np.concatenate([x_train, x_test], axis=0)
y_data = np.concatenate([y_train, y_test], axis=0).flatten()

# Reshape the data for UMAP (flatten images to 1D)
x_data_flat = x_data.reshape(x_data.shape[0], -1)

# Initialize UMAP for 3D reduction
reducer = umap.UMAP(n_components=3, random_state=42)

# Reduce dimensions for the full dataset (50,000 points)
embedding_full_3d = reducer.fit_transform(x_data_flat)

# Reduce dimensions for a subset (1,000 points)
subset_indices = np.random.choice(len(x_data_flat), 1000, replace=False)
embedding_subset_3d = reducer.fit_transform(x_data_flat[subset_indices])
y_subset = y_data[subset_indices]

# Plotting function for 3D UMAP
def plot_umap_3d(embedding, labels, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2], 
        c=labels, cmap="tab10", s=5, alpha=0.7
    )
    fig.colorbar(scatter, ax=ax, label="Classes")
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the 3D plot
    plt.show()

# Plot and save full dataset in 3D
plot_umap_3d(embedding_full_3d, y_data, "3D UMAP Projection of CIFAR-10 (50,000 points)", "umap_full_3d.png")

# Plot and save subset in 3D
plot_umap_3d(embedding_subset_3d, y_subset, "3D UMAP Projection of CIFAR-10 (1,000 points)", "umap_subset_3d.png")

