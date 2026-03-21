import logging

import numpy as np

logger = logging.getLogger("pretrain")


class CosineScheduler:
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
    ):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros(freeze_iters)

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def build_schedulers(cfg):
    lr = dict(
        base_value=cfg.lr,
        final_value=cfg.min_lr,
        total_iters=cfg.max_steps,
        warmup_iters=cfg.warmup_steps,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.weight_decay,
        final_value=cfg.weight_decay_end,
        total_iters=cfg.max_steps,
    )
    momentum = dict(
        base_value=cfg.momentum_teacher,
        final_value=cfg.final_momentum_teacher,
        total_iters=cfg.max_steps,
    )
    teacher_temp = dict(
        base_value=cfg.teacher_temp,
        final_value=cfg.teacher_temp,
        total_iters=cfg.warmup_teacher_temp_steps,
        warmup_iters=cfg.warmup_teacher_temp_steps,
        start_warmup_value=cfg.warmup_teacher_temp,
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[: cfg.freeze_last_layer_steps] = (
        0  # mimicking the original schedules
    )

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )
