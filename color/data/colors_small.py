import os
import json
import re

import pandas as pd


def _get_dataset_path():
    source_path = os.path.dirname(os.path.abspath(__file__))
    datasets_root_path = os.path.join(source_path, '..', '..', 'datasets')
    dataset_path = os.path.join(datasets_root_path, 'color-dict-small.json')
    return dataset_path


def load_color_names(max_words=None):
    with open(_get_dataset_path(), 'r') as x:
        small = json.load(x)
    colors = []
    for colorPage in small['colorPage']:
        for entry in colorPage['colorEntry']:
            colors.append(entry)

    # Standardize
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

    names = colors['name'].apply(lambda n: n.lower().strip())
    rgb = colors[['r', 'g', 'b']].astype('int')
    return names, rgb.values
