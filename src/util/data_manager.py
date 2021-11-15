import sys
import os
from pathlib import Path

import pandas as pd

sys.path.append('src')

import config as cf


def load_csv_data(path):
    file_path = os.path.join(path)

    return pd.read_csv(file_path)