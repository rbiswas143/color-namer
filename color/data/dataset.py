import os
import pickle
from collections import namedtuple
import numpy as np
import torch
import torch.utils.data as D

from log_utils import log
import color.data.embeddings as emb_data
import color.data.colors_small as colors_small
import color.data.colors_big as colors_big
import color.utils.utils as utils


def to_embeddings(sentence, vocab_dict, all_emb):
    """
    Covert a sentence to an Ndarray of word embeddings
    Words in the sentence are looked up in the vocab dictionary for their index, which is then used
    to fetch the corresponding embeddings. mbeddings for out of vocab words are initialized randomly
    :param sentence: String of space delimited words
    :param vocab_dict: Word to index mapping dictionary
    :param all_emb: Embedding matrix
    :return: Ndarray of computed embedding for the sentence. Shape (num_words, embedding_length)
    """

    # Allocate embedding Ndarray for the sentence
    words = sentence.split()
    filtered = np.ndarray((len(words), all_emb.shape[1]), dtype=np.float)

    for i, w in enumerate(words):
        if w in vocab_dict:
            # Word found in vocab. Lookup embedding
            filtered[i] = all_emb[vocab_dict[w]]
        else:
            # Assign random embedding to out-of-vocab word
            # Random embedding should have a similar distribution to the embedding matrix
            filtered[i] = np.random.rand(all_emb.shape[1]) - 0.5

    return filtered


def load_dataset_params(save_dir):
    """
    Loads the parameters that were used to create the dataset along with the created partitions
    :param save_dir: Where the datset was saved
    :return: (params dictionary, array of partitions)
    """

    # Save params
    full_path = os.path.join(save_dir, 'dataset_params.pickle')
    with open(full_path, 'rb') as x:
        params = pickle.load(x)

    # Load partitions
    partitions = []
    for partition_path in ['train_partition.txt', 'cv_partition.txt', 'test_partition.txt']:
        full_path = os.path.join(save_dir, partition_path)
        with open(full_path, 'r') as x:
            partitions.append([line.strip() for line in x])

    return params, partitions


class Dataset(D.Dataset):
    """Represents a trainable dataset and encapsulates all pre-processing tasks"""

    def __init__(self, **kwargs):

        # Dataset configuration
        self.params = {
            'dataset': 'big',  # Color dataset type: 'small', 'big'
            'emb_len': 50,  # Glove embedding length: 50, 100, 200, 300

            'normalize_rgb': True,  # Should the rgb values be normalized
            'max_words': None,  # Restrict no of words in colors dataset
            'pad_len': None,  # To train variable length sequences in batches, we can pad them to same length
            'add_stop_word': False,  # Stop word embedding is added to each color name embedding and embedding matrix

            'create_partitions': True,  # Whether to split the dataset into train, cv, and test sets
            'cv_split': 0.1,  # Fraction of the dataset to be used for Cross Validation
            'test_split': 0,  # Fraction of the dataset to be added to the Test Set
        }
        utils.dict_update_existing(self.params, kwargs)

        # Load colors dataset
        log.info('Loading colors dataset')
        colors_ds = colors_small if self.params['dataset'] == 'small' else colors_big
        self.color_names, self.color_rgb = colors_ds.load_color_names(max_words=self.params['max_words'])
        log.debug('Colors loaded. Dataset dimensions: %s', self.color_rgb.shape)

        # Process colors dataset
        if self.params['normalize_rgb']:
            log.debug('Normalizing colors')
            self.color_rgb = self.color_rgb / 256
        else:
            # Ensure dtype is double
            self.color_rgb = self.color_rgb.astype(np.float64)

        # Load embeddings
        log.info('Loading embeddings')
        self.vocab, self.embeddings = emb_data.load_embeddings(self.params['emb_len'])
        if self.params['add_stop_word']:
            self.vocab[self.vocab.index[-1] + 1] = 'STOP_WORD'
            self.embeddings = np.concatenate((self.embeddings, np.ones((1, self.params['emb_len']))), axis=0)
        log.debug('Embeddings loaded. Embedding Dimensions: %s', self.embeddings.shape)

        # Create vocab dictionary, a word to index mapping for each word in the embedding vocab
        self.vocab_dict = {v: i for i, v in enumerate(self.vocab)}

        # Convert each color name string to corresponding embedding
        self.color_name_embs = [
            to_embeddings(name, self.vocab_dict, self.embeddings)
            for name in self.color_names
        ]

        # Append stop word to each embedding
        if self.params['add_stop_word']:
            self.color_name_embs = [
                np.concatenate((emb, np.ones((1, self.params['emb_len']))), axis=0)
                for emb in self.color_name_embs
            ]

        # Pad embeddings to specified length
        if self.params['pad_len'] is not None:
            log.info('Padding embeddings for color names')
            for i in range(len(self.color_name_embs)):

                # Determine embedding length
                emb_len = self.color_name_embs[i].shape[0]
                if emb_len > self.params['pad_len']:
                    # Embedding length should not be greater than pad length
                    raise Exception(
                        'Embedding length [{}] > Padding Length [{}]'.format(emb_len, self.params['pad_len']))

                # Determine number of padding words needed
                pad_len = self.params['pad_len'] - emb_len

                # Pad with zeros
                if pad_len > 0:
                    self.color_name_embs[i] = np.pad(self.color_name_embs[i], ((0, pad_len), (0, 0)), 'constant')

        # Create train, cv and test partitions
        if self.params['create_partitions']:
            log.info('Splitting dataset')
            self.train_set, self.cv_set, self.test_set = self._split_dataset()
            log.debug('Dataset Split: Train(%d), CV(%d), Test(%d)',
                      len(self.train_set), len(self.cv_set), len(self.test_set))
        else:
            log.info('Random partitions were not created')

    def __len__(self):
        return len(self.color_rgb)

    def __getitem__(self, idx):
        """Tuple of rgb value, color name as embedding, color name as string"""
        return self.color_rgb[idx], self.color_name_embs[idx], self.color_names[int(idx)]

    def _split_dataset(self):
        cv = self.params['cv_split']
        test = self.params['test_split']
        assert cv >= 0 and test >= 0 and cv + test <= 1
        cv_len = int(len(self) * cv)
        test_len = int(len(self) * test)
        train_len = len(self) - cv_len - test_len
        return D.random_split(self, (train_len, cv_len, test_len))

    def save(self, save_dir):
        """Persist the dataset parameters and partitions"""

        # Save params
        full_path = os.path.join(save_dir, 'dataset_params.pickle')
        with open(full_path, 'wb') as x:
            pickle.dump(self.params, x)

        # Each partition is persisted saved as a dataset index and color name pair
        # Dataset index can be used to re-create the partitons
        color_idx_lookup = {name: idx for idx, (_, _, name) in enumerate(self)}
        for partition, partition_path in [
            (self.train_set, 'train_partition.txt'),
            (self.cv_set, 'cv_partition.txt'),
            (self.test_set, 'test_partition.txt'),
        ]:
            full_path = os.path.join(save_dir, partition_path)
            color_names = ['{},{}'.format(color_idx_lookup[name], name) for _, _, name in partition]
            with open(full_path, 'w') as x:
                x.write('\n'.join(color_names))

    @staticmethod
    def load(save_dir):
        """Recreate dataset from persisted parameters and partitions"""

        # Load params
        full_path = os.path.join(save_dir, 'dataset_params.pickle')
        with open(full_path, 'rb') as x:
            params = pickle.load(x)

        # Load color index for each persisted partiton partition
        partitions = []
        for partition_path in ['train_partition.txt', 'cv_partition.txt', 'test_partition.txt']:
            full_path = os.path.join(save_dir, partition_path)
            with open(full_path, 'r') as x:
                partitions.append([int(line.strip().split(',')[0]) for line in x])
        Partitions = namedtuple('Partitions', ('train', 'cv', 'test'))
        partitions = Partitions(*partitions)

        # Re-create dataset and partitions
        params['create_partitions'] = False  # Skip creating partitions randomly
        dataset = Dataset(**params)
        dataset.train_set = D.Subset(dataset, partitions.train)
        dataset.cv_set = D.Subset(dataset, partitions.cv)
        dataset.test_set = D.Subset(dataset, partitions.test)

        return dataset


class DataLoader(D.DataLoader):
    """
    Any modifications to data batches go here
    seq_len_first: True puts sequence size first in each batch, whereas the default is batch size
    """

    def __init__(self, *args, seq_len_first=False, use_cuda=False, **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)
        self.seq_len_first = seq_len_first
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def __iter__(self):
        for rgb, embs, names in super(DataLoader, self).__iter__():

            # Sequence models expect sequence length to be the first dimension
            if self.seq_len_first:
                embs = embs.view(embs.shape[1], embs.shape[0], *embs.shape[2:])

            # Move tensors to CPU/GPU
            rgb = rgb.to(self.device)
            embs = embs.to(self.device)

            yield rgb, embs, names
