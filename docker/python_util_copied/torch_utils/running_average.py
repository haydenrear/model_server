import torch


def running_average(window_length):
    count = 0
    running_avg = torch.zeros(1)

    def inner(new_value):
        nonlocal count, running_avg
        count += 1
        count = min(count, window_length)
        running_avg = running_avg * ((count - 1) / count) + new_value / count
        return running_avg

    return inner
