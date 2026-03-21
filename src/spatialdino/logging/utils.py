# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import datetime
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Generator
from pathlib import Path
from typing import Any, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


class MetricLogger:
    def __init__(
        self,
        logger: logging.Logger,
        delimiter: str = "\t",
        output_file: Union[str, Path, None] = None,
    ) -> None:
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        if output_file is not None:
            output_file = (
                Path(output_file) if isinstance(output_file, str) else output_file
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = output_file
        self.logger = logger

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = float(v.item())
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter!s}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: Any) -> None:
        self.meters[name] = meter

    def dump_in_output_file(
        self, iteration: int, iter_time: float, data_time: float, rank: int
    ) -> dict[str, Any]:
        if self.output_file is None or rank != 0:
            return {}
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: float(v.median) for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")
        return dict_to_dump

    def log_every(
        self,
        iterable: DataLoader,
        print_freq: int,
        header: str | None = None,
        n_iterations: int | None = None,
        start_iteration: int = 0,
        rank: int = 0,
    ) -> Generator[dict[str, Any], None, None]:
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(
                    iteration=i,
                    iter_time=iter_time.avg,
                    data_time=data_time.avg,
                    rank=rank,
                )
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            f"{header} Total time: {total_time_str} ({total_time / n_iterations:.6f} s / it)"
        )


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, num: int = 1) -> None:
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self) -> None:
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not dist.is_available() and dist.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> float:
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self) -> float:
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
