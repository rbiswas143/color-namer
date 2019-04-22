import os
import json

import pandas as pd


def _get_dataset_path():
    source_path = os.path.dirname(os.path.abspath(__file__))
    datasets_root_path = os.path.join(source_path, '..', '..', 'datasets')
    dataset_path = os.path.join(datasets_root_path, 'color-dict-big.json')
    return dataset_path


def load_color_names():
    with open(_get_dataset_path(), 'r') as x:
        big = json.load(x)

    # Standardize
    colors = pd.io.json.json_normalize(list(big['colors']))
    colors = colors.drop(columns=['hex', 'luminance'])
    colors = colors.rename(columns={'rgb.r': 'r', 'rgb.g': 'g', 'rgb.b': 'b'})

    names = colors['name'].apply(lambda n: n.lower())
    rgb = colors[['r', 'g', 'b']].astype('int')
    return names, rgb.values
