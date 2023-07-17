import heapq
import itertools
import os
import random
from functools import partial
from typing import Callable, Iterator, List, TypeVar

import numpy as np
import torch
from pytorch_lightning.utilities.seed import isolate_rng, seed_everything
from torch import distributed as dist
from torch.utils.data import DataChunk
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import (BatchSampler, RandomSampler,
                                      SequentialSampler, SubsetRandomSampler)
from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import (
    _clean_files, _create_dataset_directory,
    _generate_iwslt_files_for_lang_and_split, _wrap_split_argument)
from torchtext.data.functional import to_map_style_dataset


@_wrap_split_argument(("train", "valid", "test"))
def ParallelDataset(
    root=".data",
    split=("train", "valid", "test"),
    language_pair=("de", "en"),
    transforms: Callable = (None, None)
):
    src_language, tgt_language = language_pair
    src_transform, tgt_transform = transforms

    file_path_by_lang_and_split = {
        src_language: {
            "train": f"train.{src_language}",
            "valid": f"valid.{src_language}",
            "test": f"test.{src_language}",
        },
        tgt_language: {
            "train": f"train.{tgt_language}",
            "valid": f"valid.{tgt_language}",
            "test": f"test.{tgt_language}",
        }
    }

    src_filename = file_path_by_lang_and_split[src_language][split]
    full_src_filepath = os.path.join(root, src_filename)

    tgt_filename = file_path_by_lang_and_split[tgt_language][split]
    full_tgt_filepath = os.path.join(root, tgt_filename)

    # src_data_dp = FileOpener(full_src_filepath, encoding="utf-8")
    # tgt_data_dp = FileOpener(full_tgt_filepath, encoding="utf-8")

    src_lines = FileOpener([full_src_filepath], encoding="utf-8").readlines(return_path=False, strip_newline=True)
    tgt_lines = FileOpener([full_tgt_filepath], encoding="utf-8").readlines(return_path=False, strip_newline=True)

    # from itertools import count
    # count_src, count_tgt = count(), count()
    # src_lines = src_lines.to_map_datapipe(
    #     key_value_fn=lambda line: (next(count_src), line))
    # print(f'source dataset size: {len(src_lines)}')
    # tgt_lines = tgt_lines.to_map_datapipe(
    #     key_value_fn=lambda line: (next(count_tgt), line))
    # print(f'target dataset size: {len(tgt_lines)}')
    # print(len(src_lines))
    
    if src_transform is not None:
        src_lines = src_lines.map(src_transform)
    if tgt_transform is not None:
        tgt_lines = tgt_lines.map(tgt_transform)

    return src_lines.zip(tgt_lines).shuffle().sharding_filter()


T_co = TypeVar("T_co", covariant=True)

def identity(example):
    return example

class MaxTokensBatchSamplerOld(BatchSampler):
    def __init__(self,
                 sampler,
                 batch_size,
                 max_tokens,
                 drop_last,
                 sort_key: Callable = None,
                 bucket_size_multiplier=100,
                 shuffle=True):
        super().__init__(sampler, batch_size, drop_last)
        self.max_tokens = max_tokens
        self.sort_key = sort_key
        self.bucket_size_multiplier = bucket_size_multiplier
        self.shuffle = shuffle

        # Not a clean solution
        self.bucket_batches = []
        self._build_buckets()

    def __iter__(self):
        self._build_buckets()
        # Iterate over buckets
        for batches, batch_sizes in self.bucket_batches:
            # Shuffle bucket-batch order
            batches = SubsetRandomSampler(batches) if self.shuffle else batches
            for batch in batches:
                # if self.shuffle:  # Shuffle inner batch
                #     random.shuffle(batch)
                yield batch  # Batch indexes [sent1_idx, sent2_idx,...]

    def __len__(self):
        return sum([len(x[0]) for x in self.bucket_batches])

    def _build_buckets(self):
        # Randomize samples
        tmp_sampler = RandomSampler(self.sampler) if self.shuffle else self.sampler

        # Split samples in N batches (or "buckets")
        tmp_sampler = BatchSampler(tmp_sampler, min(self.batch_size * self.bucket_size_multiplier, len(self.sampler)), False)

        # Sort samples
        self.bucket_batches = []
        for bucket in tmp_sampler:
            bucket_sorted = sorted([(i, self.sort_key(i)) for i in bucket], key=lambda x: x[1])

            # Create batches constrained
            batches = []
            batch_sizes = []

            last_batch = []
            last_batch_size = 0
            for i, (sample_i, length_i) in enumerate(bucket_sorted):
                if (last_batch_size + length_i) < self.max_tokens:
                    last_batch.append(sample_i)
                    last_batch_size += length_i
                else:
                    # Add batch
                    batches.append(last_batch)
                    batch_sizes.append(last_batch_size)

                    # Add new sample
                    last_batch = [sample_i]
                    last_batch_size = length_i

            # Add last batch
            batches.append(last_batch)
            batch_sizes.append(last_batch_size)

            # Add bucket batches
            self.bucket_batches.append((batches, batch_sizes))

