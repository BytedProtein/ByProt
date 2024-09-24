import itertools
import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
import subprocess


def parse_pmpnn_args(args):
    if os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        print(40 * "-")
        print("chain_id_jsonl is NOT loaded")

    if os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        print(40 * "-")
        print("fixed_positions_jsonl is NOT loaded")
        fixed_positions_dict = None

    if os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, "r") as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        print(40 * "-")
        print("tied_positions_jsonl is NOT loaded")
        tied_positions_dict = None

    # if os.path.isfile(args.pssm_jsonl):
    #     with open(args.pssm_jsonl, "r") as json_file:
    #         json_list = list(json_file)
    #     pssm_dict = {}
    #     for json_str in json_list:
    #         pssm_dict.update(json.loads(json_str))
    # else:
    #     print(40 * "-")
    #     print("pssm_jsonl is NOT loaded")
    #     pssm_dict = None

    # if os.path.isfile(args.omit_AA_jsonl):
    #     with open(args.omit_AA_jsonl, "r") as json_file:
    #         json_list = list(json_file)
    #     for json_str in json_list:
    #         omit_AA_dict = json.loads(json_str)
    # else:
    #     print(40 * "-")
    #     print("omit_AA_jsonl is NOT loaded")
    #     omit_AA_dict = None

    # if os.path.isfile(args.bias_AA_jsonl):
    #     with open(args.bias_AA_jsonl, "r") as json_file:
    #         json_list = list(json_file)
    #     for json_str in json_list:
    #         bias_AA_dict = json.loads(json_str)
    # else:
    #     print(40 * "-")
    #     print("bias_AA_jsonl is NOT loaded")
    #     bias_AA_dict = None

    # if os.path.isfile(args.bias_by_res_jsonl):
    #     with open(args.bias_by_res_jsonl, "r") as json_file:
    #         json_list = list(json_file)

    #     for json_str in json_list:
    #         bias_by_res_dict = json.loads(json_str)
    #     print("bias by residue dictionary is loaded")
    # else:
    #     print(40 * "-")
    #     print("bias by residue dictionary is not loaded, or not provided")
    #     bias_by_res_dict = None

    return StructureDataset(args.jsonl_path, truncate=None, max_length=200000), {
        "chain_id_dict": chain_id_dict,
        "fixed_positions_dict": fixed_positions_dict,
        "tied_positions_dict": tied_positions_dict,
        "pssm_dict": None,
        "omit_AA_dict": None,
        "bias_AA_dict": None,
        "bias_by_res_dict": None,
    }


def featurize(batch, feats_dict, alphabet, device):
    (
        chain_id_dict,
        fixed_positions_dict,
        tied_positions_dict,
        omit_AA_dict,
        pssm_dict,
        bias_AA_dict,
        bias_by_res_dict,
    ) = feats_dict.values()

    (
        X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        chain_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        dihedral_mask,
        tied_pos_list_of_lists_list,
        pssm_coef,
        pssm_bias,
        pssm_log_odds_all,
        bias_by_res_all,
        tied_beta,
        seqs
    ) = tied_featurize(
        batch,
        device,
        alphabet,
        chain_id_dict,
        fixed_positions_dict,
        omit_AA_dict,
        tied_positions_dict,
        pssm_dict,
        bias_by_res_dict,
        ca_only=False,
    )

    new_batch = dict(
        names=[b["name"] for b in batch],
        seqs=seqs,
        coords=X,
        tokens=S,
        coord_mask=mask.bool(),
        lengths=lengths,
        chain_mask=chain_M.bool(),  # 1 for positions in masked_chains (chains to predict), 0 for visible chains (chains as context)
        residue_idx=residue_idx,
        # mask_otherchain=mask_self,  # m_ij = 1 where i and j not in the same chain, m_ij = 0 for being in the same chain

        chain_idx=chain_encoding_all,  # chain idx that each residue belongs to
        fixed_pos_mask=~(chain_M_pos.bool()),
        tied_pos_list=tied_pos_list_of_lists_list,
        chain_list_list=chain_list_list,
        visible_list_list=visible_list_list,
        masked_list_list=masked_list_list,
        masked_chain_length_list_list=masked_chain_length_list_list,
    )

    return new_batch


###################


class StructureDataset:
    def __init__(
        self,
        jsonl_file,
        verbose=True,
        truncate=None,
        max_length=100,
        alphabet="ACDEFGHIKLMNPQRSTVWYX-",
    ):
        alphabet_set = set([a for a in alphabet])
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry["seq"]
                name = entry["name"]

                # Convert raw coords to np arrays
                # for key, val in entry['coords'].items():
                #    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry["seq"]) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count["bad_seq_length"] += 1
                    else:
                        discard_count["too_long"] += 1
                else:
                    print(name, bad_chars, entry["seq"])
                    discard_count["bad_chars"] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print(
                        "{} entries ({} loaded) in {:.1f} s".format(
                            len(self.data), i + 1, elapsed
                        )
                    )

            print("discarded", discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tied_featurize(
    batch,
    device,
    alphabet,
    chain_dict,
    fixed_position_dict=None,
    omit_AA_dict=None,
    tied_positions_dict=None,
    pssm_dict=None,
    bias_by_res_dict=None,
    ca_only=False,
):
    """Pack and pad batch into torch tensors"""
    # alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    lengths = np.array(
        [len(b["seq"]) for b in batch], dtype=np.int32
    )  # sum of chain seq lengths
    L_max = max([len(b["seq"]) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros(
        [B, L_max], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones(
        [B, L_max, 21], dtype=np.float32
    )  # 1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    seqs = []
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    # shuffle all chains before the main loop
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[
                b["name"]
            ]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == "seq_chain_"]
            visible_chains = []
        num_chains = b["num_of_chains"]
        all_chains = masked_chains + visible_chains
        # random.shuffle(all_chains)
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                if ca_only:
                    x_chain = np.array(
                        chain_coords[f"CA_chain_{letter}"]
                    )  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack(
                        [
                            chain_coords[c]
                            for c in [
                                f"N_chain_{letter}",
                                f"CA_chain_{letter}",
                                f"C_chain_{letter}",
                                f"O_chain_{letter}",
                            ]
                        ],
                        1,
                    )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f"seq_chain_{letter}"]
                chain_seq = "".join([a if a != "-" else "X" for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked
                if ca_only:
                    x_chain = np.array(
                        chain_coords[f"CA_chain_{letter}"]
                    )  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack(
                        [
                            chain_coords[c]
                            for c in [
                                f"N_chain_{letter}",
                                f"CA_chain_{letter}",
                                f"C_chain_{letter}",
                                f"O_chain_{letter}",
                            ]
                        ],
                        1,
                    )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict != None:
                    fixed_pos_list = fixed_position_dict[b["name"]][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict != None:
                    for item in omit_AA_dict[b["name"]][letter]:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array(
                            [
                                np.argwhere(np.array(list(alphabet)) == AA)[0][0]
                                for AA in item[1]
                            ]
                        ).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b["name"]][letter]:
                        pssm_coef = pssm_dict[b["name"]][letter]["pssm_coef"]
                        pssm_bias = pssm_dict[b["name"]][letter]["pssm_bias"]
                        pssm_log_odds = pssm_dict[b["name"]][letter]["pssm_log_odds"]
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b["name"]][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict != None:
            tied_pos_list = tied_positions_dict[b["name"]]
            if tied_pos_list:
                set_chains_tied = set(
                    list(itertools.chain(*[list(item) for item in tied_pos_list]))
                )
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[
                            np.argwhere(letter_list_np == k)[0][0]
                        ]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(
                                    start_idx + v[0][v_count] - 1
                                )  # make 0 to be the first
                                tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx + v_ - 1)  # make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        seqs.append(all_sequence)
        m = np.concatenate(
            chain_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(
            fixed_position_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(
            pssm_coef_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(
            pssm_bias_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(
            pssm_log_odds_list, 0
        )  # [L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(
            bias_by_res_list, 0
        )  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(
            x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], "constant", constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]], "constant", constant_values=(0.0,))
        omit_AA_mask_pad = np.pad(
            np.concatenate(omit_AA_mask_list, 0),
            [[0, L_max - l]],
            "constant",
            constant_values=(0.0,),
        )
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(
            chain_encoding, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(
            pssm_coef_, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        pssm_bias_pad = np.pad(
            pssm_bias_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        pssm_log_odds_pad = np.pad(
            pssm_log_odds_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(
            bias_by_res_, [[0, L_max - l], [0, 0]], "constant", constant_values=(0.0,)
        )
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        # indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        indices = np.asarray([alphabet.get_idx(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(
        dtype=torch.float32, device=device
    )

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1
    )  # [B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device
    )
    lengths = torch.from_numpy(lengths).to(dtype=torch.long, device=device)
    if ca_only:
        X_out = X[:, :, 0]
    else:
        X_out = X
    return (
        X_out,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        letter_list_list,
        visible_list_list,
        masked_list_list,
        masked_chain_length_list_list,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        dihedral_mask,
        tied_pos_list_of_lists_list,
        pssm_coef_all,
        pssm_bias_all,
        pssm_log_odds_all,
        bias_by_res_all,
        tied_beta,
        seqs
    )
