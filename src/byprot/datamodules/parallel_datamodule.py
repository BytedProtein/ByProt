from functools import partial
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from byprot import utils
from byprot.datamodules.datasets import transforms, vocab
from byprot.datamodules.datasets.parallel_dataset import (
    ParallelDataset,
    to_map_style_dataset,
)
from byprot.datamodules.datasets.data_utils import MaxTokensBatchSampler

log = utils.get_logger(__name__)

def collate_batch(
    batch,
    padding_idx=0
):
    to_tensor = transforms.ToTensor(padding_value=padding_idx)

    _src, _tgt = zip(*batch)

    src = to_tensor(list(_src))
    tgt = to_tensor(list(_tgt))

    return {
        'src': src, 
        'src_padding_mask': src.eq(padding_idx),
        'tgt': tgt,
        'tgt_padding_mask': tgt.eq(padding_idx),
    }
    


class ParallelDataModule(LightningDataModule):
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
        source_lang: str = None,
        target_lang: str = None,
        joined_vocabulary: bool = False,
        batch_size: int = 64,
        max_tokens: int = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: str = 'train',
        valid_split: str = 'valid',
        test_split: str = 'test'
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.vocab_src, self.vocab_tgt = None, None
        self.load_vocab()


        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    @property
    def vocab_sizes(self) -> int:
        return len(self.vocab_src), len(self.vocab_tgt)

    @property
    def source_lang(self):
        return self.hparams.source_lang

    @property
    def target_lang(self):
        return self.hparams.target_lang

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)
        pass

    @property
    def transforms(self):
        # data transformations
        return (
            transforms.Compose(
                transforms.PlainTokenizer(),
                transforms.VocabTransform(self.vocab_src),
                transforms.AddToken(self.vocab_src.bos, begin=True),
                transforms.AddToken(self.vocab_src.eos, begin=False),
            ),
            transforms.Compose(
                transforms.PlainTokenizer(),
                transforms.VocabTransform(self.vocab_tgt),
                transforms.AddToken(self.vocab_tgt.bos, begin=True),
                transforms.AddToken(self.vocab_tgt.eos, begin=False),
            )
        )

    def load_vocab(self):
        # build vocab
        if not self.vocab_src and not self.vocab_tgt:
            vocab_src = vocab.load_vocab(self.hparams.data_dir, lang=self.source_lang)
            vocab_tgt = vocab.load_vocab(self.hparams.data_dir, lang=self.target_lang)

            if not vocab_src or not vocab_tgt:
                # ! building vocabs from raw data! do not pass transforms
                _train, _valid = ParallelDataset(
                    self.hparams.data_dir,
                    split=(self.hparams.train_split, self.hparams.valid_split),
                    language_pair=(self.source_lang, self.target_lang),
                )

                if self.hparams.joined_vocabulary:
                    log.info(f"Building joined vocabulary for [{self.source_lang}] and [{self.target_lang}] from training and valid data...")
                    vocab_src = vocab_tgt = vocab.build_vocab_from_datasets([_train, _valid], index=(0, 1))

                    vocab.save_vocab(vocab_src, self.hparams.data_dir, lang=self.source_lang)
                    vocab.save_vocab(vocab_tgt, self.hparams.data_dir, lang=self.target_lang)
                else:
                    log.info(f"Building vocabulary for [{self.source_lang}] from training and valid data...")
                    vocab_src = vocab.build_vocab_from_datasets([_train, _valid], index=0)
                    vocab.save_vocab(vocab_src, self.hparams.data_dir, lang=self.source_lang)

                    log.info(f"Building vocabulary for [{self.target_lang}] from training and valid data...")
                    vocab_tgt = vocab.build_vocab_from_datasets([_train, _valid], index=1)
                    vocab.save_vocab(vocab_tgt, self.hparams.data_dir, lang=self.target_lang)

            self.vocab_src = vocab_src
            self.vocab_tgt = vocab_tgt
            
            log.info(f"Loaded vocabulary for [{self.source_lang}] with size: {len(vocab_src)}.")
            log.info(f"Loaded vocabulary for [{self.target_lang}] with size: {len(vocab_tgt)}.")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if stage == 'fit':
            train, valid = ParallelDataset(
                self.hparams.data_dir,
                split=(self.hparams.train_split, self.hparams.valid_split),
                language_pair=(self.source_lang, self.target_lang),
                transforms=self.transforms)
            self.train_data = to_map_style_dataset(train)
            self.valid_data = to_map_style_dataset(valid)
        elif stage == 'test' or stage == 'predict':
            test = ParallelDataset(
                self.hparams.data_dir,
                split=(self.hparams.test_split),
                language_pair=(self.source_lang, self.target_lang),
                transforms=self.transforms)
            self.test_data = to_map_style_dataset(test)
        else:
            raise ValueError(f"Invalid stage: {stage}.")

    def _batch_sampler(self, dataset, max_tokens, shuffle=False, distributed=True):
        is_distributed = distributed and torch.distributed.is_initialized()

        batch_sampler = MaxTokensBatchSampler(
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            batch_size=self.hparams.batch_size, 
            max_tokens=max_tokens, 
            drop_last=False, 
            sort_key=lambda i: len(dataset[i][0]))
        return batch_sampler

    def train_dataloader(self):
        self.train_sampler = self._batch_sampler(
            self.train_data, max_tokens=self.hparams.max_tokens, shuffle=True)
        return DataLoader(
            dataset=self.train_data,
            batch_sampler=self.train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(collate_batch, padding_idx=self.vocab_src.pad)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_sampler=self._batch_sampler(
                self.valid_data, max_tokens=self.hparams.max_tokens, distributed=True),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(collate_batch, padding_idx=self.vocab_src.pad)
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_sampler=self._batch_sampler(
                self.test_data, max_tokens=self.hparams.max_tokens, distributed=True),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=partial(collate_batch, padding_idx=self.vocab_src.pad)
        )
