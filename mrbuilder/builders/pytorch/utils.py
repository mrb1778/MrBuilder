from torch import nn as nn
from torch.nn import functional as F

from mrbuilder.utils import getattr_ignore_case


def get_activation_fn(activation: str, functional=True):
    fun = getattr_ignore_case(
        F if functional else nn,
        activation)

    # todo: handle dim and inputs to activation fun, copied from functional.py
    if fun == F.softmax:
        return lambda input_: F.softmax(input_, dim=0 if input_.dim() in (0, 1, 3) else 1)
    elif fun == F.sigmoid:  # deprecated, todo: handle removal
        return lambda input_: input_.sigmoid()
    else:
        return fun
