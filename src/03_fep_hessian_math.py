import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import euclidean

def calculate_latent_precision(latent_vectors, n_components=50):
    """
    Calculates the Hessian (Precision Matrix) of a latent vector cluster 
    under the Free Energy Principle.
    """
    # Step 1: Dimensionality Reduction to prevent singular matrices
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(latent_vectors)
    
    # Step 2: Ledoit-Wolf Shrinkage for robust covariance estimation
    lw = LedoitWolf()
    cov_matrix = lw.fit(reduced_vectors).covariance_
    
    # Step 3: The Hessian is the inverse of the covariance matrix (Precision)
    hessian_matrix = np.linalg.inv(cov_matrix)
    
    # Step 4: Calculate the scalar precision (determinant of the Hessian)
    # Using log-determinant to prevent underflow in high dimensions
    sign, log_det = np.linalg.slogdet(hessian_matrix)
    scalar_precision = sign * np.exp(log_det)
    
    return hessian_matrix, scalar_precision

def calculate_cognitive_distance(learner_vector, native_centroid):
    """Calculates Euclidean distance in the entangled latent space."""
    return euclidean(learner_vector, native_centroid)
