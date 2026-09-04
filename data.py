import numpy as np
import random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import copy
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Batch
import os
import torch.nn as nn

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts))  
                        if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch


class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, seq_dict, dataset, index, args, 
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, augment_eps = 0, training = False, device="cpu"):
        
        super(ProteinGraphDataset, self).__init__()

        self.IdPairs = []
        index = set(index)
        for i in index:
            self.IdPairs.append(dataset[i])

        self.seq_dict = copy.copy(seq_dict)

        self.dataset_path = os.path.join(args.dataset_path, args.dataset) + '/'
        self.feature_path = args.feature_path
        self.dataset_name = args.dataset
        #self.task_list = task_list

        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.augment_eps = augment_eps
        self.training = training
        self.device = device
        
        #self.node_counts = [len(self.dataset[ID][0]) for ID in self.IdPairs]    # num of residues
        # if "paratope" in self.dataset_name:
        #     self.vj = torch.load(
        #         self.feature_path + 'VJsequence/' + self.dataset_name + '/' + name + ".tensor")
        # else:
        self.vj = None
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

        
    def __len__(self): return len(self.IdPairs)
    
    def __getitem__(self, idx):
        # vj = None
        return self._featurize_as_graph(idx, 0), self._featurize_as_graph(idx, 1)

    def _featurize_as_graph(self, idx, pair_idx,vj = None):
        name = self.IdPairs[idx][pair_idx]
        
        if len(self.IdPairs[idx]) <= 2:
            y = None
        else:
            y = torch.tensor([int(self.IdPairs[idx][2])],dtype=torch.float32).unsqueeze(-1)
        if "Ab" in name:
            vj = torch.load(
                self.feature_path + 'VJsequence/' + self.dataset_name + '/' + name + ".tensor")
        else:
            vj = None
        with torch.no_grad():
            coords = torch.load(self.dataset_path + "pdb/" + name + ".tensor")
            # Data augmentation
            if self.training and self.augment_eps > 0:
                coords = coords + self.augment_eps * torch.randn_like(coords)

            #seq = torch.as_tensor([self.letter_to_num[aa] for aa in self.seq_dict[name]],
            #                      device=self.device, dtype=torch.long)
            seq = None

            X_ca = coords[:, 1]     # equal to coords[:,1,:], extract coords of CA ATOM
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            edge_index, _ = add_self_loops(edge_index, num_nodes=X_ca.size(0))
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            geo_edge_feat = self._get_geo_edge_feat(X_ca, edge_index)
            # print(name)
            dihedrals = self._dihedrals(coords)
            # print("dihedrals({}): ".format(dihedrals.shape))

            # prottrans_feat = torch.load(self.feature_path + "ProtTrans/" + self.dataset_name + '/' + name + ".tensor")
            # print("prottrans_feat({}): ".format(prottrans_feat.shape))
            dssp_feat = torch.load(self.feature_path + 'DSSP/' + self.dataset_name + '/' + name + ".tensor")
            # print("dssp_feat({}): ".format(dssp_feat.shape))
            pssm_feat = torch.load(self.feature_path + 'PSSM/' + self.dataset_name + '/' + name + '.tensor')
            # print("pssm_feat({}): ".format(pssm_feat.shape))
            language_feat = torch.load(self.feature_path + "LLM/" + self.dataset_name + '/' + name + ".tensor")

            node_feat = torch.cat([dihedrals, language_feat, dssp_feat, pssm_feat], dim=-1)
            edge_feat = torch.cat([rbf, pos_embeddings, geo_edge_feat], dim=-1)

            if self.training and self.augment_eps > 0:
                node_feat = node_feat + 0.1 * self.augment_eps * torch.randn_like(node_feat)

            node_feat, edge_feat = map(torch.nan_to_num, (node_feat, edge_feat))


        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                         node_feat=node_feat, edge_feat=edge_feat,
                                         edge_index=edge_index,
                                         vj=vj,
                                         y=y)
        return data

    # def _vjsequence(self,idx,pair_idx):
    #     name = self.IdPairs[idx][pair_idx]
    #     vj = torch.load(
    #         self.feature_path + 'VJsequence/' + self.dataset_name + '/' + name + ".tensor")
    #     return vj

                                
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index, num_embeddings=None, period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _get_geo_edge_feat(self, X_ca, edge_index):
        u = torch.ones_like(X_ca)
        u[1:] = X_ca[1:] - X_ca[:-1]
        u = F.normalize(u, dim=-1)
        b = torch.ones_like(X_ca)
        b[:-1] = u[:-1] - u[1:]
        b = F.normalize(b, dim=-1)
        n = torch.ones_like(X_ca)
        n[:-1] = torch.cross(u[:-1], u[1:])
        n = F.normalize(n, dim=-1)

        local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3]

        node_j, node_i = edge_index
        t = F.normalize(X_ca[node_j] - X_ca[node_i], dim=-1)
        t = torch.einsum('ijk,ij->ik', local_frame[node_i], t) # [E, 3]
        #r = torch.sum(local_frame[node_i] * local_frame[node_j], dim=1)
        r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
        Q = self._quaternions(r) # [E, 4]

        return torch.cat([t, Q], dim=-1) # [E, 3 + 4]

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q


