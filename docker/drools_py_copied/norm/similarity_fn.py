import numpy as np
import scipy
from numpy.linalg import norm


def cosine(embedding_1, embedding_2):
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    return 1 - np.dot(embedding_1, embedding_2) / (norm(embedding_1) * norm(embedding_2))

def mahalanobis(x, means, covariances):
    # Calculates the minimum Mahalanobis distance from x to the components of a GMM
    distances = [scipy.spatial.distance.mahalanobis(x, mean, np.linalg.inv(cov))
                 for mean, cov in zip(means, covariances)]
    return min(distances)
