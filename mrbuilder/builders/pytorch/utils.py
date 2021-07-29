from torch import nn as nn
from torch.nn import functional as F

from mrbuilder.utils import getattr_ignore_case


def get_activation_fn(activation: str, functional=True):
    return getattr_ignore_case(
        F if functional else nn,
        activation)
