import argparse
import glob
import logging
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import pmpnn_utils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from byprot import utils
from byprot.datamodules.datasets import DataProcessor as PDBDataProcessor
from byprot.models.fixedbb.generator import IterativeRefinementGenerator
from byprot.utils import io
from byprot.utils.config import compose_config as Cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from collections import namedtuple

GenOut = namedtuple("GenOut", ["output_tokens", "output_scores", "attentions"])


def setup_generation(args, ckpt):
    pl.seed_everything(args.seed)

    pl_module, exp_cfg = utils.load_from_experiment(args.experiment_path, ckpt=ckpt)
    model = pl_module.model
    alphabet = pl_module.alphabet
    collater = alphabet.featurize
    generator = IterativeRefinementGenerator(
        alphabet=alphabet,
        max_iter=args.max_iter,
        strategy=args.strategy,
        temperature=args.temperature,
    )
    return model.eval(), alphabet, collater, generator


def _fill_mask(target_tokens, fixed_pos_mask, chain_mask, alphabet):
    target_mask = (
        target_tokens.ne(alphabet.padding_idx)  # & mask
        & target_tokens.ne(alphabet.cls_idx)
        & target_tokens.ne(alphabet.eos_idx)
        & ~fixed_pos_mask
        & chain_mask
    )
    _tokens = target_tokens.masked_fill(target_mask, alphabet.mask_idx)
    _mask = _tokens.eq(alphabet.mask_idx)
    return _tokens, _mask


def prepare_data(prot_feats, feats_dict, alphabet, num_seqs, device):
    prot_feats_clone = [deepcopy(prot_feats) for i in range(num_seqs)]
    batch = pmpnn_utils.featurize(prot_feats_clone, feats_dict, alphabet, device=device)
    prev_tokens, prev_token_mask = _fill_mask(batch["tokens"], batch["fixed_pos_mask"], batch["chain_mask"], alphabet)
    batch["prev_tokens"] = prev_tokens
    batch["prev_token_mask"] = prev_token_mask
    batch = utils.recursive_to(batch, device=device)
    return batch, prot_feats["seq"]


def generate(args):
    model, alphabet, collater, generator = setup_generation(args, args.ckpt)
    model = model.cuda()
    device = next(model.parameters()).device

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    dataset, feats_dict = pmpnn_utils.parse_pmpnn_args(args)

    st = time.time()
    pbar = tqdm(dataset)
    recs = []
    for protein in pbar:
        pdb_id = protein["name"]
        fp_saveto_fasta = open(os.path.join(args.out_dir, f"{pdb_id}.fasta"), 'w')
        pbar.set_description_str(f"{pdb_id}")

        batch, native_seq = prepare_data(
            protein, feats_dict, alphabet, num_seqs=args.num_seqs, device=device
        )

        with autocast():
            outputs = model.sample(
                batch=batch,
                max_iter=args.max_iter,
                strategy=args.strategy,
                temperature=args.temperature,
            )
        output_tokens = outputs[0]

        # print('final:')
        # pprint(alphabet.decode(output_tokens, remove_special=False))

        for idx, seq in enumerate(alphabet.decode(output_tokens, remove_special=True)):
            masked_chain_length_list = batch["masked_chain_length_list_list"][idx]
            masked_list = batch["masked_list_list"][idx]

            def _get_designed_seq(seq):
                start = 0
                end = 0
                list_of_seq = []
                for mask_l in masked_chain_length_list:
                    end += mask_l
                    list_of_seq.append(seq[start:end])
                    start = end

                seq = "".join(list(np.array(list_of_seq)[np.argsort(masked_list)]))
                l0 = 0
                for mc_length in list(
                    np.array(masked_chain_length_list)[np.argsort(masked_list)]
                )[:-1]:
                    l0 += mc_length
                    seq = seq[:l0] + "/" + seq[l0:]
                    l0 += 1
                return seq

            if idx == 0:
                native_seq = _get_designed_seq(native_seq)
                fixed_chains = batch["visible_list_list"][0]
                designed_chains = batch["masked_list_list"][0]
                fp_saveto_fasta.write(
                    f">{pdb_id}, fixed_chains={fixed_chains}, designed_chains={designed_chains}, seed={args.seed}\n"
                )
                fp_saveto_fasta.write(f"{native_seq}\n")
            seq = _get_designed_seq(seq)

            rec = np.mean(
                np.array([(a == b) for a, b in zip(native_seq, seq)])
            )
            recs.append(rec)

            fp_saveto_fasta.write(f">{pdb_id}: seq_{idx}, recovery={rec}\n")
            fp_saveto_fasta.write(f"{seq}\n")

        fp_saveto_fasta.close()
    print(f"Eta: {time.time() - st}. AAR: {np.mean(recs)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seqs", type=int, default=20)
    parser.add_argument("--experiment_path", type=str)
    parser.add_argument("--ckpt", type=str, default="best.ckpt")
    parser.add_argument("--pdb_dir", type=str, default="./pdbs")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--strategy", type=str, default="denoise")
    parser.add_argument("--max_iter", type=int, default=5)

    parser.add_argument(
        "--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl"
    )
    parser.add_argument(
        "--chain_id_jsonl",
        type=str,
        default="",
        help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.",
    )
    parser.add_argument(
        "--fixed_positions_jsonl",
        type=str,
        default="",
        help="Path to a dictionary with fixed positions",
    )
    parser.add_argument(
        "--tied_positions_jsonl",
        type=str,
        default="",
        help="Path to a dictionary with tied positions",
    )

    # parser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    # parser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")

    # parser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.")
    # parser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    # parser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    # parser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    # parser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    # parser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    # parser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")

    args = parser.parse_args()
    pprint(args)

    generate(args)


if __name__ == "__main__":
    main()
