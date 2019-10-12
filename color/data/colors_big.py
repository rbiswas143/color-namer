import os
import json

import pandas as pd


def _get_dataset_path():
    # Dataset path is computed relative to the location of this script
    source_path = os.path.dirname(os.path.abspath(__file__))
    datasets_root_path = os.path.join(source_path, '..', '..', 'datasets')
    dataset_path = os.path.join(datasets_root_path, 'color-dict-big.json')
    return dataset_path


def load_color_names(max_words=None):
    """
    Load big colors dataset
    :param max_words: int; In case you want to filter out colors with long names
    :return: (color names as a Series, color rgb values as an Ndarray)
    """

    # Fetch dataset path and load colors
    with open(_get_dataset_path(), 'r') as x:
        big = json.load(x)

    # Transform dataset to keep relevant fields and standardize its structure
    colors = pd.io.json.json_normalize(list(big['colors']))
    colors = colors.drop(columns=['hex', 'luminance'])
    colors = colors.rename(columns={'rgb.r': 'r', 'rgb.g': 'g', 'rgb.b': 'b'})

    # Filter out long names
    if max_words is not None:
        num_words = colors['name'].apply(lambda c: len(c.split()))
        colors = colors[num_words <= max_words].reset_index()

    # Separate names and rgb values
    names = colors['name'].apply(lambda n: n.lower().strip())
    rgb = colors[['r', 'g', 'b']].astype('int')
    return names, rgb.values
