import functools
import os
import random
from collections.abc import Sequence
from functools import lru_cache
from numbers import Number
from typing import Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import ListConfig


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    os.environ["WDS_SEED"] = str(seed)


def make_3tuple(
    x: Union[Number, tuple[Number, Number, Number], ListConfig],
) -> tuple[Number, Number, Number]:
    if isinstance(x, Sequence) and len(x) == 3:
        return tuple(x)

    assert isinstance(x, Number)
    return (x, x, x)


def make_2tuple(
    x: Union[Number, tuple[Number, Number], ListConfig],
) -> tuple[Number, Number]:
    if isinstance(x, Sequence) and len(x) == 2:
        return tuple(x)

    assert isinstance(x, Number)
    return (x, x)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def rhasattr(obj, attr):
    return rgetattr(obj, attr) is not None


@lru_cache(maxsize=128)
def find_lcm(x: int, y: int) -> int:
    if x > y:
        greater = x
    else:
        greater = y
    while True:
        if greater % x == 0 and greater % y == 0:
            lcm = greater
            break
        greater += 1
    return lcm
