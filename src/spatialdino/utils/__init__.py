import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Set, Union

import torch
import torch.amp
import torch.nn as nn

from spatialdino.distributed import save_on_master

from .utils import (
    PCA,
    cosine_dist_func,
    kmeans_fit_predict,
    # generate_centroids,
)
