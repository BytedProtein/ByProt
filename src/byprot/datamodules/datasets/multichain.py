import csv
import logging
import os
import pickle
import random
import time

import joblib
import lmdb
import numpy as np
import torch
from Bio import PDB
from Bio.PDB import PDBExceptions
from byprot import utils
from dateutil import parser
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.auto import tqdm

from .data_utils import Alphabet

log = utils.get_logger(__name__)


def parallelize(iterable, fn, desc='job', **kwds):
    data_list = joblib.Parallel(
        n_jobs=max(int(joblib.cpu_count() * 0.8), 1),
    )(
        joblib.delayed(fn)(task)
        for task in tqdm(iterable, dynamic_ncols=True, desc=desc)
    )
    return data_list


def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
    if debug:
        val_ids = []
        test_ids = []

    # read & clean list.csv
    # with open(params['LIST'], 'r') as f:
    #     reader = csv.reader(f)
    #     next(reader)
    #     rows = [[r[0], r[3], int(r[4])] for r in tqdm(reader, desc='split')
    #             if float(r[2]) <= params['RESCUT'] and
    #             parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

    import pandas as pd
    df = pd.read_csv(params['LIST'])
    df = df[
        (df['RESOLUTION'] <= params['RESCUT'])
        & (pd.to_datetime(df['DEPOSITION']) <= parser.parse(params['DATCUT']))
        # & df['SEQUENCE'].apply(lambda seq: len(seq) <= params['MAXLEN'])
    ]
    df_filtered = df.loc[:, ['CHAINID', 'RESOLUTION', 'CLUSTER']]
    rows = df_filtered.values.tolist()

    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:200]
    for r in tqdm(rows, desc='Spliting data clusters'):
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid = train
    return train, valid, test


class StructureDataset2(Dataset):

    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB

    def __init__(self,
                 split,
                 structure_dir,
                 truncate=None,
                 max_length=100,
                 rescut=3.5,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX',
                 transform=None,
                 ):
        self.alphabet_set = set([a for a in alphabet])
        self.max_length = max_length
        # self.args = args
        self.structure_dir = structure_dir
        self.rescut = rescut
        self.truncate = truncate
        self.transform = transform

        self.db_conn = None
        self.db_ids = None
        self._load_structures(split)

    def _load(self, pdb_dict_list):
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(self.alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= self.max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if self.truncate is not None and len(self.data) == self.truncate:
                return

            if self.verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

        total_count = i + 1
        log.info(f'Loaded data size: {len(self.data)}/{total_count}. Discarded: {discard_count}.')

    @property
    def _cache_db_path(self):
        return os.path.join(self.structure_dir, 'structure_cache.lmdb')

    def _connect_db(self):
        self._close_db()
        self.db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db_conn.begin() as txn:
            keys = [k.decode() for k in txn.cursor().iternext(values=False)]
            self.db_ids = keys

    def _close_db(self):
        if self.db_conn is not None:
            self.db_conn.close()
        self.db_conn = None
        self.db_ids = None

    def __load_structures(self, reset):
        all_pdbs = []
        for fname in os.listdir(self.structure_dir):
            if not fname.endswith('.pdb'):
                continue
            all_pdbs.append(fname)

        if reset or not os.path.exists(self._cache_db_path):
            todo_pdbs = all_pdbs
        else:
            self._connect_db()
            processed_pdbs = self.db_ids
            self._close_db()
            todo_pdbs = list(set(all_pdbs) - set(processed_pdbs))

        if len(todo_pdbs) > 0:
            self._preprocess_structures(todo_pdbs)

    def _load_structures(self, split, params=None, load_params=None):
        # if data_path is None:
        data_path = self.structure_dir
        if params is None:
            params = {
                "LIST": f"{data_path}/list.csv",
                "VAL": f"{data_path}/valid_clusters.txt",
                "TEST": f"{data_path}/test_clusters.txt",
                "DIR": f"{data_path}",
                "DATCUT": "2030-Jan-01",
                "RESCUT": self.rescut,  # resolution cutoff for PDBs
                "HOMO": 0.70  # min seq.id. to detect homo chains
            }
        if load_params is None:
            load_params = {
                'batch_size': 1,
                'shuffle': True,
                'pin_memory': False,
                'num_workers': 4}
        pdb_dataset = PDB_dataset2(list(split.keys()), loader_pdb, split, params)
        pdb_loader = torch.utils.data.DataLoader(pdb_dataset, worker_init_fn=worker_init_fn, **load_params)

        self._preprocess_structures(pdb_loader, params)

    def _preprocess_structures(self, pdb_loader, params):
        # tasks = []
        # for pdb_fname in pdb_loader:
        #     pdb_path = os.path.join(self.structure_dir, pdb_fname)
        #     tasks.append({
        #         'id': pdb_fname,
        #         'pdb_path': pdb_path,
        #     })

        data_list = joblib.Parallel(
            n_jobs=max(int(joblib.cpu_count() * 0.8), 1),
        )(
            joblib.delayed(parse_pdb)(
                task={
                    'entry': entry,
                    'max_length': self.max_length,
                    'params': params
                }
            )
            for entry, _ in tqdm(pdb_loader, dynamic_ncols=True, desc='Parse PDBs')
        )

        db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                id = data['name']
                ids.append(id)
                txn.put(id.encode('utf-8'), pickle.dumps(data))

    def __len__(self):
        return len(self.db_ids)

    def __getitem__(self, index):
        self._connect_db()
        id = self.db_ids[index]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        if self.transform is not None:
            data = self.transform(data)
        return data


class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX'):

        alphabet_set = alphabet
        # alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

        total_count = i + 1
        log.info(f'Loaded data size: {len(self.data)}/{total_count}. Discarded: {discard_count}.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
                 collate_fn=lambda x: x, drop_last=False):
        self.collate_fn = collate_fn
        self.dataset = dataset
        self.shuffle = shuffle
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths, kind='stable')

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [ix], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            if self.collate_fn is not None:
                batch = self.collate_fn(batch)
            yield batch


def worker_init_fn(worker_id):
    # np.random.seed()
    utils.seed_everything(seed=worker_id)


init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
extra_alphabet = [str(item) for item in list(np.arange(300))]
chain_alphabet = init_alphabet + extra_alphabet


def parse_pdb(task):
    entry = task['entry']

    t = loader_pdb(entry, task['params'])

    # chain_alphabet = task['chain_alphabet']
    max_length = task['max_length']
    # num_units = task['num_units']
    # id = task['id']

    parsed = None

    # t = {k: v[0] for k, v in t.items()}
    # c1 += 1
    if 'label' in list(t):
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        mask_list = []
        visible_list = []
        if len(list(np.unique(t['idx']))) < 352:
            for idx in list(np.unique(t['idx'])):
                letter = chain_alphabet[idx]
                res = np.argwhere(t['idx'] == idx)
                initial_sequence = "".join(list(np.array(list(t['seq']))[res][0, ]))
                if initial_sequence[-6:] == "HHHHHH":
                    res = res[:, :-6]
                if initial_sequence[0:6] == "HHHHHH":
                    res = res[:, 6:]
                if initial_sequence[-7:-1] == "HHHHHH":
                    res = res[:, :-7]
                if initial_sequence[-8:-2] == "HHHHHH":
                    res = res[:, :-8]
                if initial_sequence[-9:-3] == "HHHHHH":
                    res = res[:, :-9]
                if initial_sequence[-10:-4] == "HHHHHH":
                    res = res[:, :-10]
                if initial_sequence[1:7] == "HHHHHH":
                    res = res[:, 7:]
                if initial_sequence[2:8] == "HHHHHH":
                    res = res[:, 8:]
                if initial_sequence[3:9] == "HHHHHH":
                    res = res[:, 9:]
                if initial_sequence[4:10] == "HHHHHH":
                    res = res[:, 10:]
                if res.shape[1] < 4:
                    pass
                else:
                    my_dict['seq_chain_' + letter] = "".join(list(np.array(list(t['seq']))[res][0, ]))
                    concat_seq += my_dict['seq_chain_' + letter]
                    if idx in t['masked']:
                        mask_list.append(letter)
                    else:
                        visible_list.append(letter)
                    coords_dict_chain = {}
                    all_atoms = np.array(t['xyz'][res, ])[0, ]  # [L, 14, 3]
                    coords_dict_chain['N_chain_' + letter] = all_atoms[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = all_atoms[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = all_atoms[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = all_atoms[:, 3, :].tolist()
                    my_dict['coords_chain_' + letter] = coords_dict_chain
            my_dict['name'] = t['label']
            my_dict['masked_list'] = mask_list
            my_dict['visible_list'] = visible_list
            my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
            my_dict['seq'] = concat_seq
            if len(concat_seq) <= max_length:
                parsed = my_dict
            # if len(pdb_dict_list) >= num_units:
            #     return None
    return parsed


def loader_pdb(item, params):
    pdbid, chid = item[0].split('_')
    PREFIX = "%s/pdb/%s/%s" % (params['DIR'], pdbid[1:3], pdbid)

    # load metadata
    if not os.path.isfile(PREFIX + ".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX + ".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a, b in zip(asmb_ids, asmb_chains)
                           if chid in b.split(',')])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates) < 1:
        chain = torch.load("%s_%s.pt" % (PREFIX, chid))
        L = len(chain['seq'])
        return {'seq': chain['seq'],
                'xyz': chain['xyz'],
                'idx': torch.zeros(L).int(),
                'masked': torch.Tensor([0]).int(),
                'label': item[0]}

    # randomly pick one assembly from candidates
    with utils.local_seed(
        seed=42, enable=params['DETERMINISTIC']
    ):
        asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids) == asmb_i)[0]

    # load relevant chains
    chains = {c: torch.load("%s_%s.pt" % (PREFIX, c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    # generate assembly
    asmb = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d' % k]
        u = xform[:, :3, :3]
        r = xform[:, :3, 3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1 & s2

        # transform selected chains
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:, None, None, :]
                asmb.update({(c, k, i): xyz_i for i, xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids == chid][0, :, 1]
    homo = set([ch_j for seqid_j, ch_j in zip(seqid, chids)
                if seqid_j > params['HOMO']])
    # stack all chains in the assembly together
    seq, xyz, idx, masked = "", [], [], []
    seq_list = []
    for counter, (k, v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        seq_list.append(chains[k[0]]['seq'])
        xyz.append(v)
        idx.append(torch.full((v.shape[0],), counter))
        if k[0] in homo:
            masked.append(counter)

    return {'seq': seq,
            'xyz': torch.cat(xyz, dim=0),
            'idx': torch.cat(idx, dim=0),
            'masked': torch.Tensor(masked).int(),
            'label': item[0]}


def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                     'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    for _ in range(repeat):
        for t in tqdm(data_loader, desc='Parsing PDB'):
            t = {k: v[0] for k, v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx'] == idx)
                        initial_sequence = "".join(list(np.array(list(t['seq']))[res][0, ]))
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:, :-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:, 6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:, :-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:, :-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:, :-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:, :-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:, 7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:, 8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:, 9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:, 10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_' + letter] = "".join(list(np.array(list(t['seq']))[res][0, ]))
                            concat_seq += my_dict['seq_chain_' + letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res, ])[0, ]  # [L, 14, 3]
                            coords_dict_chain['N_chain_' + letter] = all_atoms[:, 0, :].tolist()
                            coords_dict_chain['CA_chain_' + letter] = all_atoms[:, 1, :].tolist()
                            coords_dict_chain['C_chain_' + letter] = all_atoms[:, 2, :].tolist()
                            coords_dict_chain['O_chain_' + letter] = all_atoms[:, 3, :].tolist()
                            my_dict['coords_chain_' + letter] = coords_dict_chain
                    my_dict['name'] = t['label']
                    my_dict['masked_list'] = mask_list
                    my_dict['visible_list'] = visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list


class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out


class PDB_dataset2(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params,
                 random_select=True):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params
        self.random_select = random_select

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        if self.random_select:
            sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        else:
            sel_idx = 0
        out = self.train_dict[ID][sel_idx]
        # print(f"ID: {ID}")
        # print(f"dict: {self.train_dict[ID]}")
        # print(f"sel_idx: {sel_idx}")
        # print(f"sel_out: {out}")
        # print('---------------------------------------\n')

        return out


def featurize_legacy(batch, alphabet='ACDEFGHIKLMNPQRSTVWYX', device='cpu', add_special=False):
    padding_idx, eos_idx, cls_idx = alphabet.padding_idx, alphabet.eos_idx, alphabet.cls_idx
    # alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    B = len(batch)

    names = [b['name'] for b in batch]
    seqs = [b['seq'] for b in batch]

    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])

    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)  # residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)  # for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)  # integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    # S = np.zeros([B, L_max], dtype=np.int32)  # sequence AAs integers
    S = np.full([B, L_max], fill_value=padding_idx, dtype=np.int32)  # sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                     'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)  # randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1)  # [chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
            elif letter in masked_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)

        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan, ))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], 'constant', constant_values=(0.0, ))
        chain_M[i, :] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i, :] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.get_idx(a) for a in all_sequence], dtype=np.int32)
        # indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    # mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.bool, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    # return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all

    out = dict(
        names=names,
        seqs=seqs,
        coords=X,
        tokens=S,
        coord_mask=mask,
        lengths=lengths,
        chain_mask=chain_M,  # 1 for positions in masked_chains (chains to predict), 0 for visible chains (chains as context)
        residue_idx=residue_idx,
        mask_otherchain=mask_self,  # m_ij = 1 where i and j not in the same chain, m_ij = 0 for being in the same chain
        chain_idx=chain_encoding_all,  # chain idx that each residue belongs to
    )
    return out


def bi_append(x, left, right=None, is_np=False):
    right = right or left
    if is_np:
        return np.concatenate([left[None], x, right[None]], axis=0)
    return [left] + x + [right]


def featurize(
    batch,
    alphabet: Alphabet,
    device='cpu',
    add_special_tokens=False,
    deterministic=False,
):
    padding_idx, eos_idx, cls_idx = alphabet.padding_idx, alphabet.eos_idx, alphabet.cls_idx

    B = len(batch)

    # names = [b['name'] for b in batch]
    names = []
    seqs = []

    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    if add_special_tokens:
        L_max += 2

    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)  # residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)  # for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)  # integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.full([B, L_max], fill_value=padding_idx, dtype=np.int32)  # sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                     'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains

        _seed = sum(ord(c) for c in b['name'])
        with utils.local_seed(_seed, enable=deterministic):
            random.shuffle(all_chains)  # randomly shuffle chain order

        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0

        m_chain = ''.join([c for c in all_chains if c in masked_chains])
        name = f"{b['name']}___{m_chain}"
        if len(visible_chains):
            v_chain = ''.join([c for c in all_chains if c in visible_chains])
            name = f"{name}_{v_chain}"
        names.append(name)

        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1)  # [chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
            elif letter in masked_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1

        all_sequence = "".join(chain_seq_list)
        seqs.append(all_sequence[:])
        indices = [alphabet.get_idx(a) for a in all_sequence]

        if add_special_tokens:
            l_c0 = len(chain_seq_list[0])
            residue_idx[i, 1:l1 + 1] = residue_idx[i, :l1]  # shift right
            residue_idx[i, l1] = residue_idx[i, l1 - 1] + 1  # set eos
            residue_idx[i, :l_c0 + 1] = np.arange(0, l_c0 + 1)  # increase res_id for first chain by 1

            indices = bi_append(indices, left=cls_idx, right=eos_idx)
            x_chain_list = bi_append(x_chain_list, np.full((1, 4, 3), fill_value=np.nan))
            chain_mask_list = bi_append(chain_mask_list, np.array([1.0]))
            chain_encoding_list = bi_append(chain_encoding_list, left=np.array([1]), right=np.array([c - 1]))

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(indices)
        # Convert to labels
        S[i, :l] = np.asarray(indices, dtype=np.int32)

        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan, ))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], 'constant', constant_values=(0.0, ))
        chain_M[i, :] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i, :] = chain_encoding_pad

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    isnan = np.isnan(X)
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    lengths = S.ne(padding_idx).long().sum(1)
    # mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.bool, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    # return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all

    out = dict(
        names=names,
        seqs=seqs,
        coords=X,
        tokens=S,
        coord_mask=mask,
        lengths=lengths,
        chain_mask=chain_M,  # 1 for positions in masked_chains (chains to predict), 0 for visible chains (chains as context)
        residue_idx=residue_idx,
        mask_otherchain=mask_self,  # m_ij = 1 where i and j not in the same chain, m_ij = 0 for being in the same chain
        chain_idx=chain_encoding_all,  # chain idx that each residue belongs to
    )
    return out


class Featurizer(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: dict, deterministic=False):
        return featurize(
            raw_batch,
            alphabet=self.alphabet,
            add_special_tokens=self.alphabet.add_special_tokens,
            deterministic=deterministic
        )