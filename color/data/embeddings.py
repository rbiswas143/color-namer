import os
import csv
import numpy as np
import pandas as pd


def _get_embeddings_path(size):
    # Embedding path is computed relative to the location of this script
    # Embedding size determines the exact name of the embedding file
    if size not in [50, 100, 200, 300]:
        raise Exception('Invalid embedding size: {}'.format(size))
    source_path = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(source_path, '..', '..', 'datasets', 'embeddings')
    embeddings_path = os.path.join(embeddings_dir, 'glove.6B.{}d.txt'.format(size))
    return embeddings_path


def load_embeddings(size=50):
    """
    Load and process Glove word embeddings
    :param size: Length of each embedding. Must be in (50, 100, 200, 300)
    :return: Words as a Series of size num_colors, Embeddings as a Float Ndarray of shape (num_colors, embedding_length)
    """
    path = _get_embeddings_path(size)
    emb_df = pd.read_csv(path, sep=' ', engine='c', encoding='utf-8', quoting=csv.QUOTE_NONE, header=None)
    vocab = emb_df[0]
    emb_df.drop(0, axis=1, inplace=True)
    return vocab, emb_df.astype('float').values
