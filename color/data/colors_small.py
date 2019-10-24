"""
1000+ colors from Sherwin Williams's collection of colors
https://images.sherwin-williams.com/content_images/sw-colors-name-csp-acb.acb
"""

import os
import json
import re

import pandas as pd


def _get_dataset_path():
    # Dataset path is computed relative to the location of this script
    source_path = os.path.dirname(os.path.abspath(__file__))
    datasets_root_path = os.path.join(source_path, '..', '..', 'datasets')
    dataset_path = os.path.join(datasets_root_path, 'color-dict-small.json')
    return dataset_path


def load_color_names(max_words=None):
    """
    Load small colors dataset
    :param max_words: int; In case you want to filter out colors with long names
    :return: (color names as a Series, color rgb values as an Ndarray)
    """

    # Fetch dataset path and load colors
    with open(_get_dataset_path(), 'r') as x:
        small = json.load(x)

    # Transform dataset to keep relevant fields and standardize its structure
    colors = []
    for colorPage in small['colorPage']:
        for entry in colorPage['colorEntry']:
            colors.append(entry)

    colors = pd.io.json.json_normalize(list(colors))
    colors = colors.rename(columns={
        'colorName': 'name',
        'RGB8.red': 'r',
        'RGB8.green': 'g',
        'RGB8.blue': 'b'
    })
    colors['name'] = colors['name'].apply(lambda n: re.sub('\(SW.*\)', '', n).strip())

    # Filter out long names
    if max_words is not None:
        num_words = colors['name'].apply(lambda c: len(c.split()))
        colors = colors[num_words <= max_words].reset_index()

    # Separate names and rgb values
    names = colors['name'].apply(lambda n: n.lower().strip())
    rgb = colors[['r', 'g', 'b']].astype('int')
    return names, rgb.values
