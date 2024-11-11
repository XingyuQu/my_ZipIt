import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BatchNorm2d
from utils import reset_bn_stats


def get_weights(model):
    """Get weights of a model as a numpy array"""
    if isinstance(model, torch.nn.Module):
        # model; exclude bn statistics
        return np.concatenate([p.data.cpu().numpy().ravel()
                               for p in model.parameters()])

    elif isinstance(model, dict):
        # state dict; include bn statistics
        weights = []
        for name, p in model.items():
            if 'num_batches_tracked' in name:
                continue
            else:
                weights.append(p.data.cpu().numpy().ravel())
        return np.concatenate(weights)


def interpolate_state_dicts(state_dict_1, state_dict_2, weight,
                            bias_norm=False):
    if not bias_norm:
        return {key: (1 - weight) * state_dict_1[key] +
                weight * state_dict_2[key] for key in state_dict_1.keys()}
    else:
        model_state = deepcopy(state_dict_1)
        height = 0
        for p_name in model_state:
            if "batches" not in p_name:
                model_state[p_name].zero_()
                if "weight" in p_name:
                    model_state[p_name].add_(state_dict_1[p_name], alpha=1.0 - weight)
                    model_state[p_name].add_(state_dict_2[p_name], alpha=weight)
                    height += 1
                if "bias" in p_name:
                    model_state[p_name].add_(state_dict_1[p_name], alpha=(1.0 - weight)**height)
                    model_state[p_name].add_(state_dict_2[p_name], alpha=weight**height)
                if "res_scale" in p_name:
                    model_state[p_name].add_(state_dict_1[p_name], alpha=1.0 - weight)
                    model_state[p_name].add_(state_dict_2[p_name], alpha=weight)
        return model_state


def calculate_models_distance(model_1, model_2):
    # TODO: improve choices of distance metrics
    w_1 = get_weights(model_1)
    w_2 = get_weights(model_2)
    distance = np.linalg.norm(w_1 - w_2)
    return distance


# https://github.com/KellerJordan/REPAIR/blob/master/notebooks/Train-Merge-REPAIR-VGG11.ipynb
class TrackLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)

    def get_stats(self):
        return (self.bn.running_mean, self.bn.running_var.sqrt())

    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1


class ResetLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)

    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std

    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)


# adds TrackLayers around every conv layer
def make_tracked_net(net, device=None, name='vgg'):
    net1 = deepcopy(net)
    if 'vgg' in name:
        for i, layer in enumerate(net1.features):
            if isinstance(layer, (nn.Conv2d)):
                net1.features[i] = TrackLayer(layer)
    elif 'resnet20' in name:
        for block_group in net1.segments:
            for block in block_group:
                block.conv1 = TrackLayer(block.conv1)
                block.conv2 = TrackLayer(block.conv2)
    else:
        raise NotImplementedError
    return net1.eval().to(device)


# adds ResetLayers around every conv layer
def make_repaired_net(net, device=None, name='vgg'):
    net1 = deepcopy(net).to(device)
    if 'vgg' in name:
        for i, layer in enumerate(net1.features):
            if isinstance(layer, (nn.Conv2d)):
                net1.features[i] = ResetLayer(layer)
    elif 'resnet20' in name:
        for block_group in net1.segments:
            for block in block_group:
                block.conv1 = ResetLayer(block.conv1)
                block.conv2 = ResetLayer(block.conv2)
    else:
        raise NotImplementedError
    return net1.eval().to(device)


def make_rescale_net(net, device=None, name='vgg'):
    net1 = deepcopy(net).to(device)
    if 'vgg' in name:
        for i, layer in enumerate(net1.features):
            if isinstance(layer, (nn.Conv2d)):
                net1.features[i] = RescaleLayer(layer)
    elif 'resnet20' in name:
        for block_group in net1.segments:
            for block in block_group:
                block.conv1 = RescaleLayer(block.conv1)
                block.conv2 = RescaleLayer(block.conv2)
    return net1.eval().to(device)


class BatchScale2d(BatchNorm2d):
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting
            #  this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization
          rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when
          buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        input_var = input.var([0, 2, 3])
        if bn_training and self.track_running_stats:
            self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * input_var
            return input / torch.sqrt(input_var[None, :, None, None] + self.eps) * self.weight[None, :, None, None]
        else:
            return input / torch.sqrt(self.running_var[None, :, None, None] + self.eps) * self.weight[None, :, None, None]


class RescaleLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = BatchScale2d(layer.out_channels)

    def set_stats(self, goal_std):
        self.bn.weight.data = goal_std

    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)


def repair(loader, model_tracked_s, model_repaired, device, alpha_s=None,
           variant='repair', average=False, factor=1, name='vgg'):
    model_tracked_s = [make_tracked_net(model, device, name) for model in model_tracked_s]
    for model in model_tracked_s:
        reset_bn_stats(model, loader)

    if variant == 'repair':
        model_repaired = make_repaired_net(model_repaired, device, name=name)
    elif variant == 'rescale':
        model_repaired = make_rescale_net(model_repaired, device, name=name)

    num = len(model_tracked_s)
    if alpha_s is None:
        alpha_s = [1/num] * num

    for layers in zip(*[model_tracked.modules() for model_tracked in model_tracked_s],
                      model_repaired.modules()):

        if not isinstance(layers[0], TrackLayer):
            continue

        # get neuronal statistics of original networks
        mu_s = [layer.get_stats()[0] for layer in layers[:-1]]
        std_s = [layer.get_stats()[1] for layer in layers[:-1]]
        # set the goal neuronal statistics for the merged network
        goal_mean = sum([alpha * mu for alpha, mu in zip(alpha_s, mu_s)])
        goal_std = sum([alpha * std for alpha, std in zip(alpha_s, std_s)])
        # print('goal_mean:', goal_mean)
        # print('goal_std:', goal_std)
        print()
        if average:
            goal_mean = torch.ones_like(goal_mean) * goal_mean.abs().mean() * goal_mean.sign()
            goal_std = torch.ones_like(goal_std) * goal_std.mean()
        if variant == 'repair':
            layers[-1].set_stats(goal_mean * factor, goal_std * factor)
        elif variant == 'rescale':
            layers[-1].set_stats(goal_std * factor)

    reset_bn_stats(model_repaired, loader)
    return model_repaired
