"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path

import numpy as np
from torch.utils.data.sampler import Sampler


LOG = logging.getLogger('main')
NO_LABEL = -100


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def relabel_dataset(dataset, labeled_dataset):
    
    labels_dict = dict(labeled_dataset.samples)

    unlabeled_idxs = []
    labeled_idxs = []

    for idx in range(len(dataset)):
        path, target = dataset.samples[idx]

        if path in labels_dict:
            # label_idx = dataset.class_to_idx[labels_dict[filename]]
            dataset.samples[idx] = path, target
            labeled_idxs.append(idx)
        else:
            dataset.samples[idx] = path, np.array([NO_LABEL]*len(dataset.classes)).astype('float32')
            unlabeled_idxs.append(idx)

    ### list.sort() is sorting in-place, i.e. list is changed
    ### new_list = sorted(list) is sorting not in-place, i.e. list is not changed, but return another new_list
    left_labeled_idxs = sorted(set(range(len(dataset.samples))) - set(unlabeled_idxs))
    ### sanity check if two labeled_idxs are equal
    def list_eq(a, b):
        return set(a) == set(b) and len(a) == len(b)
    assert list_eq(labeled_idxs, left_labeled_idxs), print('ERROR in relabeling dataset!')

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, unlabeled_indices, labeled_indices, batch_size, labeled_batch_size):
        self.unlabeled_indices = unlabeled_indices              ### unlabeled_idxs
        self.labeled_indices = labeled_indices          ### labeled_idxs
        self.labeled_batch_size = labeled_batch_size    ### labeled_batch_size
        self.unlabeled_batch_size = batch_size - labeled_batch_size

        assert len(self.unlabeled_indices) >= self.unlabeled_batch_size > 0
        assert len(self.labeled_indices) >= self.labeled_batch_size > 0

    def __iter__(self):
        unlabeled_iter = iterate_once(self.unlabeled_indices)
        labeled_iter = iterate_eternally(self.labeled_indices)

        ### batch are tuple, which can be called by indices
        return (
            unlabeled_batch + labeled_batch
            for (unlabeled_batch, labeled_batch)
            in  zip(grouper(unlabeled_iter, self.unlabeled_batch_size),
                    grouper(labeled_iter, self.labeled_batch_size))
        )

    def __len__(self):
        return len(self.unlabeled_indices) // self.unlabeled_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)          ### return array


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
