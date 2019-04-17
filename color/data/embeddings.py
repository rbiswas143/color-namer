import os
import numpy as np


def _get_embeddings_path(size):
    if size not in [50, 100, 200, 300]:
        raise Exception('Invalid embedding size: {}'.format(size))
    source_path = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(source_path, '..', '..', 'datasets', 'embeddings')
    embeddings_path = os.path.join(embeddings_dir, 'glove.6B.{}d.txt'.format(size))
    return embeddings_path


def load_embeddings(size=50):
    path = _get_embeddings_path(size)
    vocab = []
    emb = []
    with open(path, 'r') as x:
        for line in x:
            comps = line.split()
            vocab.append(comps[0])
            emb.append(comps[1:])
    return vocab, np.array(emb)