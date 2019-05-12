import os
import json

import pandas as pd


def _get_dataset_path():
    source_path = os.path.dirname(os.path.abspath(__file__))
    datasets_root_path = os.path.join(source_path, '..', '..', 'datasets')
    dataset_path = os.path.join(datasets_root_path, 'color-dict-big.json')
    return dataset_path


def load_color_names(max_words=None):
    with open(_get_dataset_path(), 'r') as x:
        big = json.load(x)

    # Standardize
    colors = pd.io.json.json_normalize(list(big['colors']))
    colors = colors.drop(columns=['hex', 'luminance'])
    colors = colors.rename(columns={'rgb.r': 'r', 'rgb.g': 'g', 'rgb.b': 'b'})

    # Filter out long names
    if max_words is not None:
        num_words = colors['name'].apply(lambda c: len(c.split()))
        colors = colors[num_words <= max_words].reset_index()

    names = colors['name'].apply(lambda n: n.lower().strip())
    rgb = colors[['r', 'g', 'b']].astype('int')
    return names, rgb.values
