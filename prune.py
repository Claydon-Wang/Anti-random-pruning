import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import BasePruningMethod
from copy import deepcopy
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sub_pruning(BasePruningMethod):


    PRUNING_TYPE = "unstructured"

    def __init__(self, prune_mask):

        self.prune_mask = prune_mask

    def compute_mask(self, t, default_mask):

        mask = self.prune_mask
        assert mask.shape == default_mask.shape
        return mask

    @classmethod
    def apply(cls, module, name, prune_mask):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            prune_mask (int or float): quantity of parameters to prune.
                
        """
        return super(Sub_pruning, cls).apply(module, name, prune_mask=prune_mask)
    


def sub_generation(net, pruning_mode = 'unstructured'):
    p_net = deepcopy(net)
    n_net = deepcopy(net)
    for (p_name, p_module), (n_name, n_module) in zip(p_net.named_modules(), n_net.named_modules()):

        # if isinstance(p_module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
        if isinstance(p_module, (torch.nn.Conv2d, torch.nn.Linear)):
            assert p_module.weight.shape == n_module.weight.shape
            # print(p_module.weight.shape, 'weight')
            weight_shape = p_module.weight.shape
            if pruning_mode == 'structured':
                sub_pruning_structured(p_module, n_module, name="weight", shape = weight_shape)
            else:
                sub_pruning_unstructured(p_module, n_module, name="weight", shape = weight_shape)

            # some model do not have bias(bias=False)
            try:
                assert p_module.bias.shape == n_module.bias.shape
                bias_shape = p_module.bias.shape
                # print(p_module.bias.shape, 'bias')
                if pruning_mode == 'structured':
                    sub_pruning_structured(p_module, n_module, name="bias", shape = bias_shape)
                else:
                    sub_pruning_unstructured(p_module, n_module, name="bias", shape = bias_shape)
            except:
                pass


    remove_parameters(p_net)
    remove_parameters(n_net)


    return p_net, n_net



def sub_pruning_unstructured(p_module, n_module, name, shape):

    mask_shape = shape
    # print(mask_shape)

    random_mask = np.random.rand(*mask_shape)

    p_random_mask = np.where(random_mask>0.5 , 1, 0)
    n_random_mask = np.where(random_mask>0.5 , 0, 1)   

    p_random_mask = torch.from_numpy(p_random_mask).to(device)
    n_random_mask = torch.from_numpy(n_random_mask).to(device)

    Sub_pruning.apply(module=p_module, name=name, prune_mask=p_random_mask)
    Sub_pruning.apply(module=n_module, name=name, prune_mask=n_random_mask)


def sub_pruning_structured(p_module, n_module, name, shape):

    mask_shape = shape

    raw_mask = np.ones([*mask_shape])

    if len(mask_shape[:-1]) > 0:
        mask = np.random.randint(2, size=[*mask_shape[:-1]])
        random_mask = raw_mask * np.expand_dims(mask, axis=-1)
   
    else:
        random_mask = np.random.rand(*mask_shape)


    p_random_mask = np.where(random_mask>0.5 , 1, 0)
    n_random_mask = np.where(random_mask>0.5 , 0, 1)     

    p_random_mask = torch.from_numpy(p_random_mask).to(device)
    n_random_mask = torch.from_numpy(n_random_mask).to(device)
    # print(p_random_mask)
    # print(n_random_mask)

    Sub_pruning.apply(module=p_module, name=name, prune_mask=p_random_mask)
    Sub_pruning.apply(module=n_module, name=name, prune_mask=n_random_mask)



def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        # elif isinstance(module, torch.nn.BatchNorm2d):
        #     try:
        #         prune.remove(module, "weight")
        #     except:
        #         pass
        #     try:
        #         prune.remove(module, "bias")
        #     except:
        #         pass

    return model




if __name__ == "__main__":


    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()

            self.conv1 = nn.Conv2d(1, 3, 5)
            self.conv2 = nn.Conv2d(3, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)  
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, int(x.nelement() / x.shape[0])) 
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = LeNet().cuda()

    p_net, n_net = sub_generation(net, pruning_mode='unstructured')
    print(net.fc3.bias, 'raw net')
    print(p_net.fc3.bias, 'child1')
    print(n_net.fc3.bias, 'child2')

    # print(n_net.fc2.weight)
    # print(n_net.state_dict().keys())
    # print(n_net.fc2.weight_mask, 'mask')
    # remove_parameters(model=n_net)
    
    # print(n_net.fc2.weight_mask, 'mask')
    # print(n_net.state_dict().keys())
