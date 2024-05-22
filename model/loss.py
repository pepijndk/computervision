import torch.nn
import torch.nn.functional as F
import torch.nn as nn
import tensorlayerx as tlx


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        return 0


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        return 0


def perceptual_loss(content_loss, adversarial_loss, alpha=1e-3):
    return content_loss + alpha * adversarial_loss


def mean_squared_error_loss(prediction, target):
    return torch.nn.MSELoss()(prediction, target)


def vgg_loss(prediction, target):
    return 0

