import imp
import json
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from byprot import utils
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.datapipes.map import SequenceWrapper
from torch.utils.data.dataset import Subset

from .datasets.data_utils import Alphabet, MaxTokensBatchSampler

log = utils.get_logger(__name__)

from byprot.datamodules import register_datamodule


@register_datamodule('TS50')
class Struct2SeqDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        chain_set_jsonl: str = 'chain_set.jsonl',
        chain_set_splits_json: str = 'chain_set_splits.json',
        max_length = 500,
        atoms: List[str] = ('N', 'CA', 'C', 'O'),
        proteinseq_toks: List = None,
        prepend_toks: List = None,
        append_toks: List = None,
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_esm_alphabet: bool = False,
        coord_pad_inf: bool = False,
        batch_size: int = 64,
        max_tokens: int = 6000,
        sort: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: str = 'train',
        valid_split: str = 'valid',
        test_split: str = 'test',
        to_sabdab_format: bool = False,
        to_pifold_format: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.alphabet = None

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.predict_data: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if stage == 'fit':
            (train, valid), alphabet = Struct2SeqDataset(
                self.hparams.data_dir,
                chain_set_jsonl=self.hparams.chain_set_jsonl,
                chain_set_splits_json=self.hparams.chain_set_splits_json,
                max_length=self.hparams.max_length,
                split=(self.hparams.train_split, self.hparams.valid_split),
            )
            self.train_data = train
            self.valid_data = valid
            self.alphabet = Alphabet(
                standard_toks=self.hparams.proteinseq_toks,
                prepend_toks=self.hparams.prepend_toks,
                append_toks=self.hparams.append_toks,
                prepend_bos=self.hparams.prepend_bos,
                append_eos=self.hparams.append_eos
            )
        elif stage == 'test' or stage == 'predict':
            test, alphabet = Struct2SeqDataset(
                self.hparams.data_dir,
                chain_set_jsonl=self.hparams.chain_set_jsonl,
                chain_set_splits_json=self.hparams.chain_set_splits_json,
                split=(self.hparams.test_split, )
            )
            self.test_data = test
            self.alphabet = Alphabet(
                standard_toks=self.hparams.proteinseq_toks,
                prepend_toks=self.hparams.prepend_toks,
                append_toks=self.hparams.append_toks,
                prepend_bos=self.hparams.prepend_bos,
                append_eos=self.hparams.append_eos
            )
        else:
            raise ValueError(f"Invalid stage: {stage}.")

        if self.hparams.use_esm_alphabet:
            import esm
            self.alphabet = esm.Alphabet.from_architecture('ESM-1b')

        self.batch_converter = CoordBatchConverter(
            self.alphabet, coord_pad_inf=self.hparams.coord_pad_inf, to_pifold_format=self.hparams.to_pifold_format)

        self.collate_batch = partial(
            collate_batch, 
            batch_converter=self.batch_converter,
            transform=ToSabdabDataFormat(self.alphabet) if self.hparams.to_sabdab_format else None
        )

    def _build_batch_sampler(self, dataset, max_tokens, shuffle=False, distributed=True):
        is_distributed = distributed and torch.distributed.is_initialized()

        batch_sampler = MaxTokensBatchSampler(
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            batch_size=self.hparams.batch_size, 
            max_tokens=max_tokens, 
            sort=self.hparams.sort,
            drop_last=False, 
            sort_key=lambda i: len(dataset[i]['seq']))
        return batch_sampler

    def train_dataloader(self):
        if not hasattr(self, 'train_batch_sampler'):
            self.train_batch_sampler = self._build_batch_sampler(
                self.train_data, 
                max_tokens=self.hparams.max_tokens,
                shuffle=True
        )
        return DataLoader(
            dataset=self.train_data,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_sampler=self._build_batch_sampler(
                self.valid_data, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_sampler=self._build_batch_sampler(
                self.test_data, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )


def Struct2SeqDataset(
    root=".data", 
    chain_set_jsonl='chain_set.jsonl',
    chain_set_splits_json='chain_set_splits.json',
    split=("train", "validation", "test"),
    truncate=None, max_length=500,
    alphabet='ACDEFGHIKLMNPQRSTVWY',
    transforms: Callable = (None, None),
    verbose=False
):
    src_transform, tgt_transform = transforms

    alphabet_set = set([a for a in alphabet])
    discard_count = {
        'bad_chars': 0,
        'too_long': 0,
    }


    chain_set_jsonl_fullpath = os.path.join(root, chain_set_jsonl)
    chain_set_splits_json_fullpath = os.path.join(root, chain_set_splits_json)
    remove_list = os.path.join(root, "ts50remove.txt")
    with open(remove_list) as f:
        lines = f.readlines()
        remove_list = lines
    remove_list = [i[:-1] for i in remove_list]
    if split == ('test', ):
        chain_set_jsonl_fullpath = os.path.join(root, "ts50.json")

        # 1) load the dataset
        with open(chain_set_jsonl_fullpath) as f:
            # NOTE: dataset is a list of mapping 
            # each mapping has columns: 
            #   name: str
            #   seq: str. sequence of amino acids
            #   coords: Dict[str, List[1d-array]]). e.g., {"N": [[0, 0, 0], [0.1, 0.1, 0.1], ..], "Ca": [...], ..}

            dataset: List[Dict] = []

            lines = f.readlines()
            for i, line in enumerate(lines):
                # entry = json.loads(line)
                entries = json.loads(line)
                for entry in entries:
                    seq = entry['seq']
                    name = entry['name']
                    # if name in remove_list:
                    #     continue

                    # Convert raw coords to np arrays
                    coords = {}
                    coords["N"] = np.array([i[0] for i in entry['coords']]).astype(np.float32)
                    coords["CA"] = np.array([i[1] for i in entry['coords']]).astype(np.float32)
                    coords["C"] = np.array([i[2] for i in entry['coords']]).astype(np.float32)
                    coords["O"] = np.array([i[3] for i in entry['coords']]).astype(np.float32)
                    entry['coords'] = coords
                    # for key, val in entry['coords'].items():
                    #     entry['coords'][key] = np.asarray(val, dtype=np.float32)

                    # Check if in alphabet
                    bad_chars = set([s for s in seq]).difference(alphabet_set)
                    if len(bad_chars) == 0:
                        if len(entry['seq']) <= max_length:
                            dataset.append(entry)
                        else:
                            discard_count['too_long'] += 1
                    else:
                        # print(name, bad_chars, entry['seq'])
                        discard_count['bad_chars'] += 1

                    if verbose and (i + 1) % 100000 == 0:
                        print('{} entries ({} loaded)'.format(len(dataset), i+1))

                    # Truncate early
                    if truncate is not None and len(dataset) == truncate:
                        break
            total_size = i

            dataset = SequenceWrapper(dataset)
            log.info(f'Loaded data size: {len(dataset)}/{total_size}. Discarded: {discard_count}.')
            # print(f'Loaded data size: {len(dataset)}/{total_size}. Discarded: {discard_count}.')

            # 2) split the dataset
            dataset_indices = {entry['name']: i for i, entry in enumerate(dataset)}
            with open(chain_set_splits_json_fullpath) as f:
                dataset_splits = json.load(f)

            # compatible with cath data
            split = ['validation' if s == 'valid' else s for s in split]
            if split == ['test']:
                dataset_splits = [
                    Subset(dataset, list(range(50)))
                ]
    else:
            # 1) load the dataset
        with open(chain_set_jsonl_fullpath) as f:
            # NOTE: dataset is a list of mapping 
            # each mapping has columns: 
            #   name: str
            #   seq: str. sequence of amino acids
            #   coords: Dict[str, List[1d-array]]). e.g., {"N": [[0, 0, 0], [0.1, 0.1, 0.1], ..], "Ca": [...], ..}

            dataset: List[Dict] = []

            lines = f.readlines()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                # entry = json.loads(entry)
                seq = entry['seq']
                name = entry['name']
                if name in remove_list:
                    continue

                # Convert raw coords to np arrays
                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val, dtype=np.float32)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        dataset.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    # print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                if verbose and (i + 1) % 100000 == 0:
                    print('{} entries ({} loaded)'.format(len(dataset), i+1))

                # Truncate early
                if truncate is not None and len(dataset) == truncate:
                    break
            total_size = i

            dataset = SequenceWrapper(dataset)
            log.info(f'Loaded data size: {len(dataset)}/{total_size}. Discarded: {discard_count}.')
            # print(f'Loaded data size: {len(dataset)}/{total_size}. Discarded: {discard_count}.')

            # 2) split the dataset
            dataset_indices = {entry['name']: i for i, entry in enumerate(dataset)}
            with open(chain_set_splits_json_fullpath) as f:
                dataset_splits = json.load(f)

            # compatible with cath data
            split = ['validation' if s == 'valid' else s for s in split]
            dataset_splits = [
                Subset(dataset, [
                    dataset_indices[chain_name] for chain_name in dataset_splits[key] 
                    if chain_name in dataset_indices
                ])
                for key in split
            ]
            sizes = [f'{split[i]}: {len(dataset_splits[i])}' for i in range(len(split))]
            msg_sizes = ', '.join(sizes)
            log.info(f'Size. {msg_sizes}')

    if len(dataset_splits) == 1:
        dataset_splits = dataset_splits[0]
    return dataset_splits, alphabet_set
