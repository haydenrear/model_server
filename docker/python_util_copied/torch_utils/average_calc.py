import torch


def average(count, running_avg, new_value):
    return running_avg * ((count - 1) / count) + new_value / count
