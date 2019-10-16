import os
import time
import numpy as np


def get_rel_path(src_path, *rel_path):
    """Construct paths from relative path components easily"""
    src_dir = os.path.dirname(os.path.abspath(src_path))
    return os.path.abspath(os.path.join(src_dir, *rel_path))


def get_unique_key(length=6):
    """Uses current time to generate a unique key of specified length"""
    assert length <= 10
    return int(time.time() % (10 ** length))


def get_trainable_params(model):
    """Counts the trainable parameters of a PyTorch model"""
    count = 0
    for param in list(model.parameters()):
        if param.requires_grad:
            count += np.prod(param.size())
    return count


def dict_update_existing(dict_a, dict_b, ignore_new=False):
    """Update only existing keys in first dictionary. This saves you from typos in dictionary keys"""
    new_keys = dict_b.keys() - dict_a.keys()
    if len(new_keys) > 0 and not ignore_new:
        raise Exception('Invalid key: {}'.format(iter(new_keys).__next__()))
    dict_a.update(dict_b)
