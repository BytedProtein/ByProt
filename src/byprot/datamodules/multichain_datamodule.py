import os
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from byprot import utils
from byprot.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .datasets.data_utils import Alphabet, MaxTokensBatchSampler
from .datasets.multichain import (PDB_dataset2, StructureDataset,
                                  StructureLoader, build_training_clusters,
                                  featurize, loader_pdb, parse_pdb,
                                  worker_init_fn)

log = utils.get_logger(__name__)


MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB


def make_dataset(
    split, args, alphabet,
    params=None,
    load_params=None,
    data_path=None,
    deterministic=False
):
    if data_path is None:
        data_path = args.data_dir
    if params is None:
        params = {
            "LIST": f"{data_path}/list.csv",
            "VAL": f"{data_path}/valid_clusters.txt",
            "TEST": f"{data_path}/test_clusters.txt",
            "DIR": f"{data_path}",
            "DATCUT": "2030-Jan-01",
            "RESCUT": args.rescut,  # resolution cutoff for PDBs
            "HOMO": 0.70,  # min seq.id. to detect homo chains
            "MAXLEN": args.max_length,
        }
    params["DETERMINISTIC"] = deterministic

    pdb_dataset = PDB_dataset2(
        list(split.keys()), loader_pdb,
        split, params,
        random_select=not deterministic
    )
    pdb_loader = torch.utils.data.DataLoader(
        pdb_dataset, batch_size=1,
        worker_init_fn=worker_init_fn,
        pin_memory=False, shuffle=not deterministic)

    pdb_dict_list = joblib.Parallel(
        n_jobs=max(int(joblib.cpu_count() * 0.8), 1),
    )(
        joblib.delayed(parse_pdb)(
            task={
                'entry': entry,
                'max_length': args.max_length,
                'params': params
            }
        )
        for entry, _ in tqdm(pdb_loader, dynamic_ncols=True, desc='Parse PDBs')
    )
    pdb_dict_list = filter(None, pdb_dict_list)

    dataset = StructureDataset(
        pdb_dict_list,
        alphabet=alphabet,
        truncate=None,
        max_length=args.max_length
    )
    return dataset


@register_datamodule('multichain')
class MultichainDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        max_length=500,
        rescut=3.5,
        atoms: List[str] = ('N', 'CA', 'C', 'O'),
        alphabet=None,
        batch_size: int = 64,
        max_tokens: int = 6000,
        sort: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: str = 'train',
        valid_split: str = 'valid',
        test_split: str = 'test',
        to_sabdab_format: bool = False,
        to_pifold_format: bool = False,
        debug=False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.alphabet = None

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None):
        t0 = time.time()

        data_path = self.hparams.data_dir
        params = {
            "LIST": f"{data_path}/list.csv",
            "VAL": f"{data_path}/valid_clusters.txt",
            "TEST": f"{data_path}/test_clusters.txt",
            "DIR": f"{data_path}",
            "DATCUT": "2030-Jan-01",
            "RESCUT": self.hparams.rescut,  # resolution cutoff for PDBs
            "HOMO": 0.70,  # min seq.id. to detect homo chains
            "MAXLEN": self.hparams.max_length
        }

        self.hparams.data_cache = f"{data_path}/cache.db"
        self.hparams.batch_size = self.hparams.max_tokens
        if self.hparams.debug:
            self.hparams.num_examples_per_epoch = 50
            self.hparams.max_protein_length = 1000
            self.hparams.batch_size = 1000

        if os.path.exists(self.hparams.data_cache):
            train, valid, test = load_cache(self.hparams.data_cache, ['train', 'valid', 'test'])
        else:
            train, valid, test = build_training_clusters(params, debug=self.hparams.debug)

        _deterministic = False
        if stage != 'fit':
            _deterministic = True

        # if self.hparams.use_esm_alphabet:
        #     self.alphabet = Alphabet(name='esm', dataset='multichain')
        # else:
        #     self.alphabet = Alphabet(name='protein_mpnn', dataset='multichain')
        self.alphabet = Alphabet(**self.hparams.alphabet)
        self.collate_fn = partial(
            self.alphabet.featurize,
            deterministic=_deterministic)

        _make_dataset = partial(
            make_dataset,
            args=self.hparams,
            params=params,
            alphabet=self.alphabet.all_toks,
            deterministic=_deterministic
        )
        if stage == 'fit':
            self.train_dataset = _make_dataset(split=train)
            self.valid_dataset = _make_dataset(split=valid)
        elif stage == 'test' or stage == 'predict':
            self.test_dataset = _make_dataset(split=test)
        else:
            raise ValueError(f"Invalid stage: {stage}.")

        log.info(f'Data loaded (elapsed {time.time() - t0}.')

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
        return StructureLoader(self.train_dataset, batch_size=self.hparams.max_tokens, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return StructureLoader(self.valid_dataset, batch_size=self.hparams.max_tokens, collate_fn=self.collate_fn, shuffle=False)

    def test_dataloader(self):
        return StructureLoader(self.test_dataset, batch_size=self.hparams.max_tokens, collate_fn=self.collate_fn, shuffle=False)
