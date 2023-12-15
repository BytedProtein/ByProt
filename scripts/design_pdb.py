import argparse
from copy import deepcopy
import glob
import logging
import os
import random
from pathlib import Path
from pprint import pprint
import time

import numpy as np
import torch
from torch.cuda.amp import autocast
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from byprot import utils
from byprot.datamodules.datasets import DataProcessor as PDBDataProcessor
from byprot.models.fixedbb.generator import IterativeRefinementGenerator
from byprot.utils import io
from byprot.utils.config import compose_config as Cfg
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from collections import namedtuple

GenOut = namedtuple(
    'GenOut', 
    ['output_tokens', 'output_scores', 'attentions']
)

def setup_generation(args, ckpt):
    pl.seed_everything(args.seed)

    pl_module, exp_cfg = utils.load_from_experiment(
        args.experiment_path, ckpt=ckpt)
    model = pl_module.model
    alphabet = pl_module.alphabet
    collater = alphabet.featurize
    generator = IterativeRefinementGenerator(
        alphabet=alphabet, 
        max_iter=args.max_iter,
        strategy=args.strategy,
        temperature=args.temperature
    )
    return model.eval(), alphabet, collater, generator


def _full_mask(target_tokens, coord_mask, alphabet):
    target_mask = (
        target_tokens.ne(alphabet.padding_idx)  # & mask
        & target_tokens.ne(alphabet.cls_idx)
        & target_tokens.ne(alphabet.eos_idx)
    )
    _tokens = target_tokens.masked_fill(
        target_mask, alphabet.mask_idx
    )
    _mask = _tokens.eq(alphabet.mask_idx) & coord_mask
    return _tokens, _mask


def prepare_data(pdb_path, alphabet, collator, num_seqs, device):
    pdb_id = Path(pdb_path).stem
    structure = PDBDataProcessor().parse_PDB(pdb_path)
    batch = collator(
        [
            deepcopy(structure) for idx in range(num_seqs)
        ]
    )
    prev_tokens, prev_token_mask = _full_mask(
        batch['tokens'], batch['coord_mask'], alphabet
    )
    batch['prev_tokens'] = prev_tokens
    batch['prev_token_mask'] = prev_tokens.eq(alphabet.mask_idx)
    batch = utils.recursive_to(batch, device=device)
    return batch, structure['seq']


def generate(args):
    model, alphabet, collater, generator = setup_generation(args, args.ckpt)
    model = model.cuda(); 
    device = next(model.parameters()).device

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    st = time.time()
    pbar = tqdm(glob.glob(f"{args.pdb_dir}/*.pdb"))
    for pdb_path in pbar:
        pdb_id = Path(pdb_path).stem
        fp_saveto_fasta = open(os.path.join(args.out_dir, f"{pdb_id}.fasta"), 'w')
        pbar.set_description_str(f"{pdb_id}")

        batch, native_seq = prepare_data(
            pdb_path, alphabet, collater, 
            num_seqs=args.num_seqs, device=device
        )

        with autocast():
            outputs = generator.generate(model=model, batch=batch)
        output_tokens = outputs[0]

        # print('final:')
        # pprint(alphabet.decode(output_tokens, remove_special=False))

        recs = []
        for idx, seq in enumerate( 
            alphabet.decode(output_tokens, remove_special=True)
        ):
            rec = np.mean([(a==b) for a, b in zip(native_seq, seq)])
            fp_saveto_fasta.write(
                f">{pdb_id}: seq_{idx}, recovery={rec}\n")
            fp_saveto_fasta.write(f"{seq}\n")
            recs.append(rec)
        fp_saveto_fasta.close()
    print(f"Eta: {time.time() - st}. AAR: {np.mean(recs)}") 
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seqs', type=int, default=20)
    parser.add_argument('--experiment_path', type=str)
    parser.add_argument('--ckpt', type=str, default='best.ckpt')
    parser.add_argument('--pdb_dir', type=str, default='./pdbs')
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--strategy', type=str, default='denoise')
    parser.add_argument('--max_iter', type=int, default=5)
    args = parser.parse_args()
    pprint(args)

    generate(args)
    

if __name__ == '__main__':
    main()
