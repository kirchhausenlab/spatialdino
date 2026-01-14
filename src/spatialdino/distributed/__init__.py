import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from torch import inf


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup(
    distributed: bool = True,
    backend: str = "cpu:gloo,cuda:nccl",
    timeout: datetime.timedelta = datetime.timedelta(minutes=30),
) -> Tuple[int, int, int]:
    if not distributed:
        return 0, 0, 1

    world_size = (
        int(os.environ["WORLD_SIZE"])
        if "WORLD_SIZE" in os.environ
        else torch.cuda.device_count()
    )
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timeout,
        world_size=world_size,
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    return local_rank, rank, world_size


def cleanup(distributed: bool = True) -> None:
    if not distributed:
        return

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
