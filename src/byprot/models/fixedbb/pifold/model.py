import time
import torch
import torch.nn as nn
# from utils import gather_nodes, _dihedrals, _get_rbf, _orientations_coarse_gl_tuple
from .modules import *
from .modules import _dihedrals, _get_rbf, _orientations_coarse_gl_tuple

import torch_geometric as pyg

class PiFoldModel(nn.Module):
    def __init__(self, args,  **kwargs):
        """ Graph labeling network """
        super().__init__()
        self.args = args
        node_features = args.node_features
        edge_features = args.edge_features
        hidden_dim = args.hidden_dim
        dropout = args.dropout
        num_encoder_layers = args.num_encoder_layers
        self.top_k = args.k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        # prior_matrix = [
        #     [-0.58273431, 0.56802827, -0.54067466],
        #     [0.0       ,  0.83867057, -0.54463904],
        #     [0.01984028, -0.78380804, -0.54183614],
        # ]

        # self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix)[:self.args.virtual_num,:])

        self.virtual_atoms = nn.Parameter(torch.rand(self.args.virtual_num,3))

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
            node_in += pair_num*self.num_rbf
        if self.args.node_angle:
            node_in += 12
        if self.args.node_direct:
            node_in += 9
        
        edge_in = 0
        if self.args.edge_dist:
            pair_num = 16
            
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
            edge_in += pair_num*self.num_rbf
        if self.args.edge_angle:
            edge_in += 4
        if self.args.edge_direct:
            edge_in += 12

        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True) 
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout)

        self.decoder = MLPDecoder(hidden_dim, vocab=args.n_vocab)
        self._init_params()

        self.encode_t = 0
        self.decode_t = 0

    def encode(self, X, mask, lengths, S=None):
        X, S, score, h_V, h_P, P_idx, batch_id, mask_bw, mask_fw, decoding_order= self._get_features(
            S=S, X=X, score=None, mask=mask, lengths=lengths)
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))

        return {
            'node_feats': h_V, 'edge_feats': h_P, 'edge_idx': P_idx, 'batch_id': batch_id, 'node_mask': mask
        }
    
    def decode(self, encoder_out, prev_tokens=None):
        h_V = encoder_out['node_feats']
        h_P = encoder_out['edge_feats']
        P_idx = encoder_out['edge_idx']
        batch_id = encoder_out['batch_id']
        max_num_nodes = encoder_out['node_mask'].shape[1]

        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)
        log_probs, logits = self.decoder(h_V, batch_id)

        logits = pyg.utils.to_dense_batch(logits, batch_id, max_num_nodes=max_num_nodes)[0]
        h_V = pyg.utils.to_dense_batch(h_V, batch_id, max_num_nodes=max_num_nodes)[0]
        return logits, h_V

    def forward(self, X, S, mask, lengths):
        t1 = time.time()

        encoder_out = self.encode(X, mask, lengths=lengths,  S=S)

        t2 = time.time()

        logits, h_V = self.decode(encoder_out)
                
        t3 = time.time()
        
        self.encode_t += t2-t1
        self.decode_t += t3-t2

        return logits, {'feats': h_V}

    def _forward(self, h_V, h_P, P_idx, batch_id, S=None, AT_test = False, mask_bw = None, mask_fw = None, decoding_order= None, return_logit=False, return_feats=False):
        t1 = time.time()
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)
        t2 = time.time()

        log_probs, logits = self.decoder(h_V, batch_id)
                
        t3 = time.time()
        

        self.encode_t += t2-t1
        self.decode_t += t3-t2

        if return_feats:
            return logits, h_V

        if return_logit == True:
            return log_probs, logits
        return log_probs
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    def _get_features(self, S, score, X, mask, lengths):
        device = X.device
        mask_bool = (mask==1)
        # mask_bool = torch.ones_like(mask, dtype=torch.bool)
        # mask_bool = length_to_mask(lengths)
        B, N, _,_ = X.shape
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x:  torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        randn = torch.rand(mask.shape, device=X.device)+5
        decoding_order = torch.argsort(-mask*(torch.abs(randn))) #我们的mask=1代表数据可用, 而protein MPP的mask=1代表数据不可用，正好相反
        mask_size = mask.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        # 计算q已知的情况下, q->p的mask, 
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend2 = torch.gather(order_mask_backward, 2, E_idx)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1])
        mask_bw = (mask_1D * mask_attend2).unsqueeze(-1)
        mask_fw = (mask_1D * (1-mask_attend2)).unsqueeze(-1)
        mask_bw = edge_mask_select(mask_bw).squeeze()
        mask_fw = edge_mask_select(mask_fw).squeeze()
        

        # sequence
        if S is not None:
            S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X, 0) 
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = X[:,:,0,:]
        atom_Ca = X[:,:,1,:]
        atom_C = X[:,:,2,:]
        atom_O = X[:,:,3,:]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        if self.args.virtual_num>0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                        + virtual_atoms[i][1] * b \
                                        + virtual_atoms[i][2] * c \
                                        + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append( node_mask_select(_get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        
        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                # # true atoms
                for j in range(0, i):
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf).squeeze()))
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()
        

        pair_lst = ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C', 'N-N', 'N-O', 'O-N', 'O-O']


        
        edge_dist = [] #Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)
            edge_dist.append(edge_mask_select(rbf))

        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))
                for j in range(0, i):
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf)))
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        h_V = []
        if self.args.node_dist:
            h_V.append(V_dist)
        if self.args.node_angle:
            h_V.append(V_angles)
        if self.args.node_direct:
            h_V.append(V_direct)
        
        h_E = []
        if self.args.edge_dist:
            h_E.append(E_dist)
        if self.args.edge_angle:
            h_E.append(E_angles)
        if self.args.edge_direct:
            h_E.append(E_direct)
        
        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)
        
        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        # shift = torch.arange(0, B, device=E_idx.device) * N
        # shift = lengths.cumsum(dim=0) - lengths
        # shift = torch.zeros_like(shift)

        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()

        decoding_order = node_mask_select((decoding_order+shift.view(-1,1)).unsqueeze(-1)).squeeze().long()
        
        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        # sparse_idx = length_to_mask(lengths).nonzero()  # index of non-zero values
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]

        return X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order


def length_to_mask(lens, maxlen=None):
    # lens: (bsz)
    maxlen = maxlen or lens.max()
    lens = lens.view(-1)
    mask = torch.arange(maxlen, device=lens.device)[None, :] < lens[:, None]
    return mask