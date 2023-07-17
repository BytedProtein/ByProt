import os
from functools import reduce
from typing import List, OrderedDict, Union

import torch
from torch import Tensor
from torchtext.vocab import Vocab as V
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab as build_vocab

from byprot import utils

log = utils.get_logger(__name__)


def _AugmentedVocab(v: V):
    class AugmentedVocab(V):
        def __init__(self, vocab):
            super().__init__(vocab)
            self.pad = self['<pad>']
            self.unk = self['<unk>']
            self.eos = self['<eos>']
            self.bos = self['<bos>']

        def encode(self, tokens: List[str]):
            return V.lookup_indices(self, tokens)

        def decode(
            self,
            indices: Union[List[int], List[List[int]]],
            remove_special=True,
            bpe_symbol='@@ ',
        ) -> List[str]:

            if isinstance(indices, Tensor):
                indices = indices.detach().cpu().tolist()

            if isinstance(indices[0], List):
                return [
                    self.decode(_indices, remove_special, bpe_symbol) for _indices in indices
                ]

            if remove_special and self.eos in indices:
                indices = indices[:indices.index(self.eos)]
            text = ' '.join(self.lookup_tokens(indices))
            return post_process(text, bpe_symbol)

    return AugmentedVocab(v.vocab)


def load_vocab(root='./data', file_prefix='vocab', lang='en'):
    """
      Expected vocab file:
      each line contains a <token - freq pair>, separated by a space
      e.g., 
      (root/vocab.en) 
      hello 42
      world 11
    """
    vocab_full_path = os.path.join(root, f"{file_prefix}.{lang}")
    if not os.path.exists(vocab_full_path):
        return None

    ordered_dict = OrderedDict()
    with open(vocab_full_path) as fp:
        for line in fp:
            token, freq = line.strip().split()
            ordered_dict[token] = int(freq)

    vocab: V = build_vocab(ordered_dict, min_freq=0)
    vocab.set_default_index(vocab["<unk>"])
    return _AugmentedVocab(vocab)


def build_vocab_from_alphabet(alphabet: Union[List[str], str], specials=[]):
    if isinstance(alphabet, str):
        alphabet = list(alphabet)

    ordered_dict = OrderedDict(
        {element: 1 for element in alphabet}
    )

    vocab: V = build_vocab(ordered_dict, min_freq=0, specials=['<pad>', '<unk>', '<eos>'] + specials)
    vocab.set_default_index(vocab["<unk>"])
    return _AugmentedVocab(vocab)


def yield_tokens(data_iter, index=0):
    if isinstance(index, int):
        index = [index]

    for from_to_tuple in data_iter:
        tokens = []
        for i in index:
            tokens.extend(from_to_tuple[i].split(' '))
        yield tokens


def build_vocab_from_datasets(datasets, index):
    datasets_concat = reduce(lambda a, b: a + b, datasets)

    vocab: V = build_vocab_from_iterator(
        yield_tokens(datasets_concat, index),
        min_freq=2, specials=["<pad>", "<unk>", "<eos>", "<bos>"],
    )
    vocab.set_default_index(vocab["<unk>"])
    return _AugmentedVocab(vocab)


def save_vocab(vocab: V, root, file_prefix='vocab', lang='en'):
    vocab_full_path = os.path.join(root, f"{file_prefix}.{lang}")
    tokens = vocab.get_itos()

    log.info(f'Saving vocabulary for {lang} to {vocab_full_path}')
    with open(vocab_full_path, 'w') as fp:
        for idx, token in enumerate(tokens):
            fp.write(f"{token} {idx}\n")


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re

        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(" +", " ", sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence
