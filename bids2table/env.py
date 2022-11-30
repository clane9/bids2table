import sys

import hydra
import numpy as np
import omegaconf
import pandas as pd
import pyarrow as pa
from tabulate import tabulate

import bids2table


def collect_env_info():
    """
    Collect information about user system.

    - platform
    - python version
    - bids2table version
    - dependency versions
    """
    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("bids2table", bids2table.__version__))
    data.append(("hydra", hydra.__version__))
    data.append(("omegaconf", omegaconf.__version__))
    data.append(("numpy", np.__version__))
    data.append(("pandas", pd.__version__))
    data.append(("pyarrow", pa.__version__))
    env_str = tabulate(data)
    return env_str
