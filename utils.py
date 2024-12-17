
import os
# from gluoncv.torch.engine.config import get_cfg_defaults
import torch




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class Normalize(torch.nn.Module):
    def __init__(self,dataset_name):
        super(Normalize, self).__init__()
        assert dataset_name in ['imagenet', 'cifar100', 'inc', 'tensorflow'], 'check dataset_name'
        self.mode = dataset_name
        if dataset_name == 'imagenet':
            self.normalize = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        elif dataset_name == 'cifar100':
            self.normalize = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        else:
            self.normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        x = input.clone()
        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        else:
            for i in range(x.shape[1]):
                x[:,i] = (x[:,i] - self.normalize[0][i]) / self.normalize[1][i]
        return x

