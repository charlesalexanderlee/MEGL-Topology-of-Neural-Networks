import tensorflow as tf
import numpy as np
import umap
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_data = np.concatenate([x_train, x_test], axis=0)
y_data = np.concatenate([y_train, y_test], axis=0).flatten()

# Reshape the data for UMAP (flatten images to 1D)
x_data_flat = x_data.reshape(x_data.shape[0], -1)

# Initialize UMAP for dimensionality reduction
reducer = umap.UMAP(random_state=42)

# Reduce dimensions for the full dataset (50,000 points)
embedding_full = reducer.fit_transform(x_data_flat)

# Reduce dimensions for a subset (1,000 points)
subset_indices = np.random.choice(len(x_data_flat), 1000, replace=False)
embedding_subset = reducer.fit_transform(x_data_flat[subset_indices])
y_subset = y_data[subset_indices]

# Plotting function
def plot_umap(embedding, labels, title, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=2, alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(10), label="Classes")
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the plot
    plt.show()

# Plot and save full dataset
plot_umap(embedding_full, y_data, "UMAP Projection of CIFAR-10 (50,000 points)", "umap_full.png")

# Plot and save subset
plot_umap(embedding_subset, y_subset, "UMAP Projection of CIFAR-10 (1,000 points)", "umap_subset.png")

# Save embeddings
np.save("umap_embedding_full.npy", embedding_full)
np.save("umap_embedding_subset.npy", embedding_subset)
