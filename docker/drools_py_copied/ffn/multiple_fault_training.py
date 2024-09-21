import random

import torch


class MultipleFaultTraining:

    def __init__(self, epochs_per_fault: int):
        self.epochs_per_fault = epochs_per_fault
        self.num_epochs = 0

    def add_fault(self, weight: torch.Tensor):
        weight_shape = weight.shape

        if len(weight_shape) < 2 and self.num_epochs <= self.epochs_per_fault:
            self.num_epochs += 1
            return

        self.num_epochs += 0

        r = random.Random()

        # if r.randint(0, 1) == 0:
        index = r.randint(0, weight_shape[0] - 1)
            # y-direction

        xavier = torch.nn.init.xavier_normal_(torch.zeros([weight_shape[0], 0]))
        weight.T.index_copy_(0, torch.tensor([4, 0]), xavier)
        #     weight.T[0:][index]
        # else:
        #     index = r.randint(0, weight_shape[1] - 1)
        #     # x-direction
        #     xavier = torch.nn.init.xavier_normal_(torch.zeros([weight_shape[1], 0]))
        #     weight[0:][index] = xavier

        return weight
