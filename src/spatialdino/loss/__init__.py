from enum import Enum


class PixelLoss(Enum):
    l1 = "l1"
    l2 = "l2"
    charbonnier = "charbonnier"


class LatentLoss(Enum):
    ce = "ce"
