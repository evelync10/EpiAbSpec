import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch.nn import functional as F
from torch_geometric.data import Batch

class TransformerLayer(nn.Module):
    def __init__(self, node_h_dim, edge_h_dim, heads=4, dropout=0.2):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(node_h_dim) for _ in range(2)])

        self.attention = TransformerConv(in_channels=node_h_dim, out_channels=int(node_h_dim / 4), heads=heads, dropout = dropout, edge_dim = edge_h_dim, root_weight=False)
        self.dense = PositionWiseFeedForward(node_h_dim, node_h_dim * 4)

    def forward(self, h_V, edge_index, h_E):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class GraphTrans_encoder(nn.Module):
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.2):
        super(GraphTrans_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20
        
        self.W_v = nn.Linear(node_in_dim, node_h_dim, bias=True)

        self.W_e = nn.Sequential(
            nn.Linear(edge_in_dim, edge_h_dim, bias=True),
            nn.LayerNorm(edge_h_dim)
        )

        self.layers = nn.ModuleList(
                TransformerLayer(node_h_dim=node_h_dim, edge_h_dim = edge_h_dim, heads=4, dropout = 0.2)
            for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq=None):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        
        return h_V
class VDJGeneEncoder(nn.Module):
    def __init__(self, vocab_size, v_embed, j_embed,dropout_rate):
        super(VDJGeneEncoder, self).__init__()

        self.v_embedding = nn.Embedding(vocab_size, v_embed)
        self.j_embedding = nn.Embedding(vocab_size, j_embed)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # print(x)
        # print(x[:,  5:10])
        h_v_embedding = self.v_embedding(x[:, 0:5])
        h_j_embedding = self.j_embedding(x[:, 5:10])
        # l_v_embedding = self.v_embedding(x[:, 10:15])
        # l_j_embedding = self.j_embedding(x[:, -5:])
        # print(h_v_embedding.shape)
        # print(h_j_embedding.shape)

        # Flatten the embeddings
        h_v_flat = h_v_embedding.view(x.size(0), -1)
        h_j_flat = h_j_embedding.view(x.size(0), -1)
        # l_v_flat = l_v_embedding.view(x.size(0), -1)
        # l_j_flat = l_j_embedding.view(x.size(0), -1)
        # print(h_v_flat.shape)
        # print(h_j_flat.shape)

        # Concatenate the flattened embeddings
        # concatenated = torch.cat([h_v_flat, h_j_flat, l_v_flat, l_j_flat], dim=-1)
        concatenated = torch.cat([h_v_flat, h_j_flat], dim=-1)
        # print(concatenated.shape)

        # Apply dropout
        output = self.dropout(concatenated)
        return output

class GraphTrans(nn.Module):
    def __init__(self, node_input_dim_0, node_input_dim_1, edge_input_dim, hidden_dim, num_layers, dropout,vocab_size, v_embed_dim,
                 j_embed_dim, vj_dropout_rate,
                 num_heads
                 ):
        super(GraphTrans, self).__init__()

        # self.node_dim_transform = nn.Linear(node_input_dim_0, node_input_dim_0)

        self.GraphTrans_encoder_0 = GraphTrans_encoder(node_in_dim=node_input_dim_0, node_h_dim=hidden_dim, edge_in_dim=edge_input_dim, edge_h_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)
        self.GraphTrans_encoder_1 = GraphTrans_encoder(node_in_dim=node_input_dim_1, node_h_dim=hidden_dim, edge_in_dim=edge_input_dim, edge_h_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)
        self.vj_gene_encoder = VDJGeneEncoder(vocab_size, v_embed_dim, j_embed_dim, vj_dropout_rate)
        pooled_dim = 2 * num_heads * hidden_dim + 5 * (v_embed_dim + j_embed_dim)
        self.add_module("FC_1", nn.Linear(pooled_dim, 2*hidden_dim, bias=True))
        self.add_module("FC_2", nn.Linear(2*hidden_dim, 1, bias=True))

        self.pool = global_mean_pool
        self.ATFC = nn.Sequential(
            nn.Linear(hidden_dim, 64)
            , nn.LeakyReLU()
            , nn.LayerNorm(64, eps=1e-6)
            , nn.Linear(64, num_heads)
        )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, data):
        # h_V, edge_index, h_E, seq
        #print("data.batch: ",data[1].batch)
        data0, data1 = data
        # print(data0.node_feat.shape)
        # print(data1.node_feat.shape)
        
        
        vj_embedding = self.vj_gene_encoder(data1.vj)
        # print("vj_embedding: ",vj_embedding.shape)
        #print("to_data_list", data0.to_data_list())
        #debug
        """ test_batch = []
        for unbatched in data0.to_data_list():
            print(unbatched)
            test_batch.append(unbatched)
        new_batch = Batch.from_data_list(test_batch)
        print("data: ", data0, "new_batch", new_batch)
        print(data0 == new_batch) """
        #print("edge_index: ",data0.edge_index.shape, data0.edge_index)
        # adjusted_node_feat_0 = self.node_dim_transform(data0.node_feat)
        # print(adjusted_node_feat_0.shape)
        h_V0 = self.GraphTrans_encoder_0(data0.node_feat, data0.edge_index, data0.edge_feat) # [num_residue, hidden_dim]
        h_V1 = self.GraphTrans_encoder_1(data1.node_feat, data1.edge_index, data1.edge_feat) # [num_residue, hidden_dim]
        # print("h_V0: ", h_V0.shape)
        # print("h_V1: ", h_V1.shape)
        h_V0_split = torch_geometric.utils.unbatch(h_V0, data0.batch)
        h_V1_split = torch_geometric.utils.unbatch(h_V1, data1.batch)
        h_V0_padded = torch.nn.utils.rnn.pad_sequence(h_V0_split, batch_first=True, padding_value=0.0)
        h_V1_padded = torch.nn.utils.rnn.pad_sequence(h_V1_split, batch_first=True, padding_value=0.0)
        # print("h_V0_padded: ", h_V0_padded.shape)
        # print("h_V1_padded: ", h_V1_padded.shape)

        # # Multi-head self-attention pooling
        att0 = self.ATFC(h_V0_padded)  # [B, L, num_heads]
        mask_zeros0 = (h_V0_padded[:, :, 1] == 0).unsqueeze(-1)
        att0 = att0.masked_fill(mask_zeros0, 1e-9)
        att0 = F.softmax(att0, dim=1)
        att0 = att0.transpose(1, 2)  # [B, num_heads, L]
        h_V_att0 = att0 @ h_V0_padded  # [B, num_heads, hidden_dim]
        h_V_att0 = torch.flatten(h_V_att0, start_dim=1)  # [B, num_heads*hidden_dim]
        # print("h_V_att0: ", h_V_att0.shape)

        att1 = self.ATFC(h_V1_padded)  # [B, L, num_heads]
        mask_zeros1 = (h_V1_padded[:, :, 1] == 0).unsqueeze(-1)
        att1 = att1.masked_fill(mask_zeros1, 1e-9)
        att1 = F.softmax(att1, dim=1)
        att1 = att1.transpose(1, 2)  # [B, num_heads, L]
        h_V_att1 = att1 @ h_V1_padded  # [B, num_heads, hidden_dim]
        h_V_att1 = torch.flatten(h_V_att1, start_dim=1)  # [B, num_heads*hidden_dim]
        # print("h_V_att1: ", h_V_att1.shape)

        # pooled_0 = self.pool(h_V0, data0.batch)
        # pooled_1 = self.pool(h_V1, data1.batch)
        # print("pooled_0: ", pooled_0.shape)
        # print("pooled_1: ", pooled_1.shape)

        pooled_s = torch.cat((h_V_att0, h_V_att1), -1)
        # print("pooled_s: ", pooled_s.shape)
        # print("pooled_vj: ", vj_embedding.shape)

        pooled = torch.cat((pooled_s, vj_embedding), -1)
        # print("pooled: ", pooled.shape)

        emb = F.elu(self._modules["FC_1"](pooled))
        output = self._modules["FC_2"](emb)

        return output
