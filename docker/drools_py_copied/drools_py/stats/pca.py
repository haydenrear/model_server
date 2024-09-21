import math

import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def calculate_reconstruction_error(original_data, reconstructed_data):
    mse = mean_squared_error(original_data, reconstructed_data)
    return mse


def pca_reconstruction(original_data, n_components):
    # Step 1: Fit PCA model and obtain explained variance ratio
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(original_data)
    explained_variance_ratio = pca.explained_variance_ratio_

    n_components = reduced_data.shape[1]  # Number of principal components
    n_data_points = len(original_data)

    # Step 2: Calculate reconstruction errors for each component
    reconstruction_errors_per_component = torch.zeros((n_data_points, n_components))

    for i in range(n_components):
        # Zero out all other components
        reduced_data_copy = reduced_data.copy()
        reduced_data_copy[:, :i] = 0
        reduced_data_copy[:, i + 1:] = 0

        # Step 3: Perform inverse transformation to reconstruct data points
        reconstructed_data = pca.inverse_transform(reduced_data_copy)

        # Step 4: Calculate reconstruction error for each data point
        reconstruction_errors = torch.tensor([calculate_reconstruction_error(original_data[j], reconstructed_data[j])
                                              for j in range(n_data_points)])

        reconstruction_errors_per_component[:, i] = reconstruction_errors

    variances = original_data.var(dim=1)
    percentage_of_reconstruction = (1 - (reconstruction_errors_per_component.T / variances)) * 100
    return percentage_of_reconstruction
