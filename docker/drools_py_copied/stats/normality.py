import logging

import numpy as np
import torch
from scipy.stats import normaltest


def calculate_normality(array):
    res, p_value = normaltest(array)
    return res, p_value


def remove_not_normal_argmin(data_set: list[float], window_size: int) -> (list[float], float, int):
    """
    When we do the PCA and determine which data points contribute to the reconstruction, then we need to find the data
    points to include. So we assume that the thing being reconstructed is a normally distributed variable, and so we
    need to decide which to include, where we want to include the biggest that contributed the most. So we start with all,
    and we start removing the lowest ones. Then after we remove each one, we check to see if it is more like a normal
    distribution or less like a normal distribution, comparing to the average change over the last window_size. If it
    is more like a normal distribution, then we continue. If it is less, then we add back the last value and return
    the most normally distributed greedy values.
    :param data_set:
    :param window_size:
    :return: the dataset after the removed, the p_value for hypothesis not normal, and the number min removed.
    """
    if len(data_set) < 35:
        logging.debug('Data set was less than 35. Non-viable cluster.')
        return data_set

    prev_normality, p_value = calculate_normality(data_set)

    rolling_window: list[float] = []

    prev = None
    num_removed = 0

    while len(data_set) >= 35:
        index_to_remove = np.argmin(data_set)

        num_removed += 1

        prev = data_set.pop(index_to_remove)

        new_normality, p_value = calculate_normality(data_set)

        rolling_window.append(new_normality)

        mean_normality = torch.mean(torch.tensor(rolling_window))

        if new_normality >= mean_normality:
            if len(rolling_window) > window_size:
                rolling_window.remove(rolling_window[0])
        elif len(rolling_window) >= window_size or len(data_set) <= 35:
            break

    if prev:
        data_set.insert(0, prev)

    if len(data_set) < 35:
        logging.debug('Data set was less than 35. Non-viable cluster.')
        return [], 0.0, num_removed

    _, p_value = calculate_normality(data_set)

    prev_p_value = p_value
    counter = 0
    while prev_p_value <= p_value and len(data_set) > 20:
        index_to_remove = np.argmin(data_set)
        data_set.pop(index_to_remove)
        prev_p_value = p_value
        _, p_value = calculate_normality(data_set)
        num_removed += 1
        counter += 1

    return data_set, p_value, num_removed
