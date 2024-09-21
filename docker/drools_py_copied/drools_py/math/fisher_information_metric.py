import numpy as np

from drools_py.math.hessian import hessian_fd


def fisher_information_metric(feature_1_log_likelihood: np.array,
                              feature_2_log_likelihood: np.array):
    return -np.mean(hessian_fd(feature_1_log_likelihood, feature_2_log_likelihood), axis=0)
