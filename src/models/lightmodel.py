import torch 
import torch.nn as nn

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import to_scipy_sparse_matrix, unbatch_edge_index
from scipy.sparse.csgraph import connected_components

import math
from typing import Union, Optional, Tuple
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

from copy import deepcopy
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import Batch
from torch_geometric.nn.inits import glorot, zeros


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = nn.Sequential(nn.Linear(in_channels, out_channels),
                       nn.ReLU(),
                       nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        tmp = torch.cat([edge_attr, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """
        return inputs
    

class GATv2ConvMOT(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        share_weights: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels)
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, bias=bias)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=bias)
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.last_projector = nn.Linear(2*out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)
        glorot(self.att)
        glorot(self.last_projector.weight)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)

        out_prop = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out_prop.view(-1, self.heads * self.out_channels)
        else:
            out = out_prop.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
    
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Attention coefficients are calculated using node features, but messages are the aggregated edge features.

        x = x_i - x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        if edge_attr is not None:
            return edge_attr * alpha.unsqueeze(-1)
        else:
            return x_j * alpha.unsqueeze(-1)
    
    def update(self, inputs: Tensor, x) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        x_l, x_r = x
        return self.last_projector(torch.cat([x_l, inputs], dim=-1))
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


def sigmoid_log_double_softmax(
        sim: torch.Tensor, logaritmic: bool) -> torch.Tensor:
    """ create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    scores = sim.new_full((b, m, n), 0)

    if logaritmic:
        scores0 = F.log_softmax(sim, 2)
        scores1 = F.log_softmax(
            sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        scores[:, :m, :n] = (scores0 + scores1)/2
    else:
        scores0 = F.softmax(sim, 2)
        scores1 = F.softmax(
            sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        scores[:, :m, :n] = (scores0 + scores1)/2
    
    return scores

class MatchAssignment(nn.Module):
    def __init__(self, dim: int, logaritmic= False) -> None:
        super().__init__()
        self.dim = dim
        self.logaritmic = logaritmic

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """ build assignment matrix from descriptors """
        n, d = x.shape
        x = x / d**.25
        
        sim = torch.full((n, n), torch.finfo(torch.float).min, device=x.device)
        sim[edge_index[0], edge_index[1]] = torch.einsum('md,nd->mn', x, x)[edge_index[0], edge_index[1]]

        scores = sigmoid_log_double_softmax(
                        sim.unsqueeze(0), 
                        logaritmic=self.logaritmic
                        )

        if self.logaritmic:
            return scores.exp().squeeze(), sim
        return scores.squeeze(), sim

class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0., use_layernorm=False, **kwargs):
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for i, dim in enumerate(fc_dims):
            layers.append(nn.Linear(input_dim, dim, **kwargs))
            if use_layernorm and dim != 1:
                layers.append(nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6))

            if i != len(fc_dims) - 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DotProductLinkPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductLinkPredictor, self).__init__()

    def forward(self, x_i, x_j):
        d = x_i.shape[1]
        x_i = x_i / d**.25
        x_j = x_j / d**.25
        out = (x_i*x_j).sum(-1)
        return torch.sigmoid(out)
    
    def reset_parameters(self):
      pass

class LightGlueMOT(nn.Module):

    def __init__(self, params):
        super(LightGlueMOT, self).__init__()
        
        self.params = params

        self.edge_enc = MLP(**params['edge_enc'])
        self.init_enc = MLP(**params['node_enc'])
        # self.node_enc = MLP(**params['node_enc'])

        self.pos_enc = PositionalEncoding(**params['pos_enc'])

        self.enc_layer = nn.TransformerEncoderLayer(**params['transformer_params'])
 
        self.encoder = nn.TransformerEncoder(self.enc_layer, num_layers=params['num_transformer_enc_layers'])

        self.joint_enc = nn.ModuleList([
            nn.TransformerEncoderLayer(**params['transformer_params'])
            for _ in range(params['num_joint_enc_layers'])
        ])
        
        self.cross_gnn = nn.ModuleList([
            GATv2ConvMOT(**params['GATv2ConvMOT_params'])
            for _ in range(params['num_joint_enc_layers'])
        ]) 
   
        self.edge_conv = nn.ModuleList([
            EdgeConv(**params['edge_conv_params'])
            for _ in range(params['num_joint_enc_layers'])
            ])
        
        self.matcher = MatchAssignment(**params['matcher_params'])
        self.classifier = MLP(**params['classifier_params'])
        
    def forward(self, graph, depth_token, depth):


        # 0) Obtain Data 
        x, x_track, edge_index, edge_features = graph.x, graph.x_track, graph.edge_index, graph.edge_features
        tracklet_mask = graph.tracklet_mask  
        bipartite_labels = graph.bipartite_labels
        mask = torch.where(bipartite_labels%2!=0, True, False).to(x.device).squeeze()
        
        if depth > 0:
            tracklet_mask[mask] = torch.flip(tracklet_mask[mask], dims=[1,])
            special_token_mask = torch.ones((x_track.shape[0],1), dtype=torch.bool, device = x_track.device)
            tracklet_mask = ~torch.cat([special_token_mask, tracklet_mask], dim=1)
        else:
            tracklet_mask = None
    
        edge_features = self.edge_enc(edge_features)
        pos_feats = self.init_enc(x_track)

        # ADD positional Embeddings.
        pos_feats[mask] = torch.flip(pos_feats[mask], dims=[1,])
        pos_feats = self.pos_enc(pos_feats.permute(1,0,2)).permute(1,0,2)

        # Concatenate interaction token with positional features.
        depth_token = depth_token.unsqueeze(0).expand(pos_feats.shape[0], -1).unsqueeze(1)
        pos_feats = torch.cat([depth_token, pos_feats], dim=1)

        # First transformer encoder layer
        pos_feats = self.encoder(pos_feats, src_key_padding_mask = tracklet_mask)
        pos_token, trans_features = pos_feats[:, 0, :].squeeze(), pos_feats[:, 1:, :]

        outputs_dict = {
            'classified_edges': [], 
            'appearance_sim': F.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1)
            }

        for idx, (gnn, edge_gnn, transformer) in enumerate(zip(self.cross_gnn, self.edge_conv, self.joint_enc)):
            
            pos_token = gnn(pos_token, edge_index, edge_features)
            edge_features = edge_gnn(pos_token, edge_index, edge_features)

            pos_feats = transformer(torch.cat([pos_token.unsqueeze(1), trans_features], dim=1),  src_key_padding_mask = tracklet_mask)
            pos_token, trans_features = pos_feats[:, 0, :].squeeze(), pos_feats[:, 1:, :]

            ## Make predictins
            pos_preds = self.classifier(
                            torch.cat(
                                [edge_features, pos_token[edge_index[0]] - pos_token[edge_index[1]]], 
                                dim=1
                                )
                            )
            
            outputs_dict['classified_edges'].append(pos_preds)
 

        return outputs_dict
