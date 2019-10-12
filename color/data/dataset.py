import os
import pickle
import numpy as np
import torch
import torch.utils.data as D
import torch.nn.functional as F

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
    :return: Ndarray of computed embedding for the sentence
    """

    # Alocate embedding Ndarray for the sentence
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

    def __init__(self, **kwargs):

        # Dataset configuration
        self.params = {
            'max_words': None,
            'dataset': 'big',  # 'small', 'big'
            'emb_len': 50,  # 50, 100, 200, 300
            'normalize_rgb': False,
            'use_cuda': False,
            'batch_size': 1,
            'cv_split': 0.1,
            'test_split': 0,
            'num_workers': 0,
            'pad_len': None,
            'var_seq_len': False,
            'add_stop_word': False
        }
        utils.dict_update_existing(self.params, kwargs)

        # Device
        if self.params['use_cuda'] and not torch.cuda.is_available():
            raise Exception('CUDA is not available')
        self.device = torch.device('cuda' if self.params['use_cuda'] else 'cpu')

        # Process colors
        log.info('Loading colors dataset')
        colors_ds = colors_small if self.params['dataset'] == 'small' else colors_big
        self.color_names, self.color_rgb = colors_ds.load_color_names(max_words=self.params['max_words'])
        log.debug('Colors loaded. Dataset dimensions: %s', self.color_rgb.shape)
        self.color_rgb = torch.tensor(self.color_rgb).float().to(self.device)
        if self.params['normalize_rgb']:
            log.debug('Normalizing colors')
            self.color_rgb /= 256

        # Process embeddings
        log.info('Loading embeddings')
        self.vocab, self.embeddings = emb_data.load_embeddings(self.params['emb_len'])
        if self.params['add_stop_word']:
            self.vocab[self.vocab.index[-1] + 1] = 'STOP_WORD'
            self.embeddings = np.concatenate((self.embeddings, np.ones((1, self.params['emb_len']))), axis=0)
        log.debug('Embeddings loaded. Embedding Dimensions: %s', self.embeddings.shape)
        self.vocab_dict = {v: i for i, v in enumerate(self.vocab)}
        self.color_name_embs = [
            torch.tensor(to_embeddings(name, self.vocab_dict, self.embeddings)).float().to(self.device)
            for name in self.color_names
        ]

        # Pad embeddings
        if self.params['pad_len'] is not None:
            log.info('Padding embeddings for color names')
            for i in range(len(self.color_name_embs)):
                emb_len = self.color_name_embs[i].shape[0]
                if emb_len > self.params['pad_len']:
                    raise Exception(
                        'Embedding length [{}] > Padding Length [{}]'.format(emb_len, self.params['pad_len']))
                pad_len = self.params['pad_len'] - emb_len
                if pad_len > 0:
                    self.color_name_embs[i] = F.pad(self.color_name_embs[i], [0, 0, 0, pad_len])

        # Data split
        log.info('Splitting dataset')
        self.train_set, self.cv_set, self.test_set = self.split()
        log.debug('Dataset Split: Train(%d), CV(%d), Test(%d)',
                  len(self.train_set), len(self.cv_set), len(self.test_set))
        self.train_loader = self.get_subset_loader(self.train_set)
        self.cv_loader = self.get_subset_loader(self.cv_set)

    def __len__(self):
        return len(self.color_rgb)

    def __getitem__(self, idx):
        return self.color_rgb[idx], self.color_name_embs[idx], self.color_names[int(idx)]

    def split(self):
        cv = self.params['cv_split']
        test = self.params['test_split']
        assert cv >= 0 and test >= 0 and cv + test <= 1
        cv_len = int(len(self) * cv)
        test_len = int(len(self) * test)
        train_len = len(self) - cv_len - test_len
        return D.random_split(self, (train_len, cv_len, test_len))

    def get_subset_loader(dataset, subset):

        class CustomDataLoader(D.DataLoader):
            def __iter__(dataloader):
                loader = super(CustomDataLoader, dataloader).__iter__()
                for rgb, embs, names in loader:
                    if dataset.params['add_stop_word']:
                        stop_emb = torch.ones((embs.shape[0], 1, embs.shape[2])).to(dataset.device)
                        embs = torch.cat((embs, stop_emb), dim=1)
                    if dataset.params['var_seq_len']:
                        embs = embs.view(embs.shape[1], embs.shape[0], embs.shape[2])
                    yield rgb, embs, names

        return CustomDataLoader(subset, batch_size=dataset.params['batch_size'],
                                shuffle=True, num_workers=dataset.params['num_workers'])

    def save(self, save_dir):
        # Save params
        full_path = os.path.join(save_dir, 'dataset_params.pickle')
        with open(full_path, 'wb') as x:
            pickle.dump(self.params, x)

        # Save partitions
        for partition, partition_path in [
            (self.train_set, 'train_partition.txt'),
            (self.cv_set, 'cv_partition.txt'),
            (self.test_set, 'test_partition.txt'),
        ]:
            full_path = os.path.join(save_dir, partition_path)
            color_names = [name for _, _, name in partition]
            with open(full_path, 'w') as x:
                x.write('\n'.join(color_names))
