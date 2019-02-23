from itertools import islice, chain

import numpy as np

from data import TwoStreamBatchSampler

def test_two_stream_batch_sampler():
    import sys
    print(sys.version)
    sampler = TwoStreamBatchSampler(unlabeled_indices=range(10),
                                    labeled_indices=range(-2, 0),
                                    batch_size=3,
                                    labeled_batch_size=1)
    print(sampler)
    batches = list(sampler)
    print(batches)

    # All batches have length 3
    assert all(len(batch) == 3 for batch in batches)

    # All batches include two items from the primary batch (unlabeled samples)
    assert all(len([i for i in batch if i >= 0]) == 2 for batch in batches)
    # first (batch_size - labeled_batch_size) samples are unlabeled
    assert all( all(np.array(batch[ : sampler.unlabeled_batch_size]) >= 0) for batch in batches)

    # All batches include one item from the secondary batch
    assert all(len([i for i in batch if i < 0]) == 1 for batch in batches)
    # last (labeled_batch_size) samples are labeled
    assert all( all(np.array(batch[-sampler.labeled_batch_size:]) < 0) for batch in batches)

    # All primary items are included in the epoch
    assert len(sampler.unlabeled_indices) % sampler.unlabeled_batch_size == 0 # Pre-condition
    assert sorted(i for i in chain(*batches) if i >= 0) == list(range(10)) # Post-condition
    print(i for i in chain(*batches))
    print(sorted(i for i in chain(*batches)))

    # Secondary items are iterated through before beginning again
    assert sorted(i for i in chain(*batches[:2]) if i < 0) == list(range(-2, 0))


def test_two_stream_batch_sampler_uneven():
    import sys
    print(sys.version)
    sampler = TwoStreamBatchSampler(unlabeled_indices=range(11),
                                    labeled_indices=range(-3, 0),
                                    batch_size=5,
                                    labeled_batch_size=2)
    print(sampler)
    batches = list(sampler)
    print(batches)

    # All batches have length 5
    assert all(len(batch) == 5 for batch in batches)

    # All batches include 3 items from the primary batch
    assert all(len([i for i in batch if i >= 0]) == 3 for batch in batches)

    # All batches include 2 items from the secondary batch
    assert all(len([i for i in batch if i < 0]) == 2 for batch in batches)

    # Almost all primary items are included in the epoch
    primary_items_met = [i for i in chain(*batches) if i >= 0]
    left_out = set(range(11)) - set(primary_items_met)
    print(left_out)
    assert len(left_out) == 11 % 3
    print(sorted(i for i in chain(*batches)))

    # Secondary items are iterated through before beginning again
    assert sorted(i for i in chain(*batches[:3]) if i < 0) == sorted(list(range(-3, 0)) * 2)


# test_two_stream_batch_sampler()
test_two_stream_batch_sampler_uneven()
