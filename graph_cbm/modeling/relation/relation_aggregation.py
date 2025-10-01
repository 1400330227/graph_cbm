import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, scatter
from torch_geometric.nn.inits import glorot, zeros

class Aggregate(nn.Module):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(Aggregate, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=bias)
            self.lin_r = self.lin_l if share_weights else nn.Linear(in_channels, heads * out_channels, bias=bias)
        else:
            self.lin_l = nn.Linear(in_channels[0], heads * out_channels, True)
            self.lin_r = self.lin_l if share_weights else nn.Linear(in_channels[1], heads * out_channels, True)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights: bool = None):

        H, C = self.heads, self.out_channels
        num_nodes = x.size(0)

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Input tensor must be 2-dimensional"
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = self.lin_r(x).view(-1, H, C) if not self.share_weights else x_l
        else:  # (Rarely used) For bipartite graphs
            x_l, x_r = x[0], x[1]
            x_l = self.lin_l(x_l).view(-1, H, C)
            x_r = self.lin_r(x_r).view(-1, H, C)

        if self.add_self_loops:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes
            )

        source_nodes, target_nodes = edge_index[0], edge_index[1]

        x_i = x_l[target_nodes]  # 目标节点特征 [num_edges, H, C]
        x_j = x_r[source_nodes]  # 源节点特征   [num_edges, H, C]

        attention_input = x_i + x_j

        if self.lin_edge is not None and edge_attr is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, H, C)
            attention_input += edge_emb

        attention_input = F.leaky_relu(attention_input, self.negative_slope)
        alpha = (attention_input * self.att).sum(dim=-1)  # [num_edges, H]
        alpha = softmax(alpha, target_nodes, num_nodes=num_nodes)

        if return_attention_weights:
            self._alpha = alpha.mean(dim=1)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        weighted_messages = x_j * alpha.unsqueeze(-1)
        out = scatter(weighted_messages, target_nodes, dim=0, dim_size=num_nodes, reduce='sum')

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if return_attention_weights:
            return out, (edge_index, self._alpha)
        else:
            return out


class RelationAggregation(nn.Module):
    def __init__(self, node_feature_dim: int, num_relations: int, num_layers: int = 2,
                 dropout_prob: float = 0.3, heads: int = 4, relation_embedding_dim: int = 64):
        super(RelationAggregation, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, relation_embedding_dim)
        self.layers = nn.ModuleList()
        hidden_channels = node_feature_dim // heads

        for index in range(num_layers):
            self.layers.append(
                Aggregate(
                    in_channels=node_feature_dim,
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout_prob,
                    edge_dim=relation_embedding_dim
                )
            )

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, node_features, edge_index_list, edge_type_list, num_nodes_list):
        if not edge_index_list or all(e.numel() == 0 for e in edge_index_list):
            return node_features

        offset = 0
        edge_index_batch = []
        edge_type_batch = []
        for i, edge_index in enumerate(edge_index_list):
            edge_index_batch.append(edge_index + offset)
            edge_type_batch.append(edge_type_list[i])
            offset += num_nodes_list[i]

        edge_index_batch = torch.cat(edge_index_batch, dim=1).contiguous()
        edge_type_batch = torch.cat(edge_type_batch, dim=0).contiguous()
        edge_attr = self.relation_embedding(edge_type_batch)

        x = node_features
        edges_attention_batch = []
        edges_index = None
        edges_attention = None
        for i, layer in enumerate(self.layers):
            x_update, attentions = layer(x, edge_index_batch, edge_attr=edge_attr, return_attention_weights=True)
            x = x + self.dropout(self.activation(x_update))

            edge_index_with_self_loops, attention_weights_full = attentions
            is_not_self_loop = edge_index_with_self_loops[0] != edge_index_with_self_loops[1]
            edges_attention = attention_weights_full[is_not_self_loop]
            edges_index = edge_index_with_self_loops[:, is_not_self_loop]

            edges_attention_batch.append(edges_attention)

        edges_attention_batch = torch.stack(edges_attention_batch, dim=0)
        return x, (edges_index, edges_attention, edges_attention_batch)
