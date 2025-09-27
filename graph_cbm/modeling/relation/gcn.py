import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class GATConv(MessagePassing):
    """
    The GATv2 operator from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the static
    attention problem of the standard :class:`~torch_geometric.conv.GATConv`
    layer: since the linear layers in the standard GAT are applied right after
    each other, the ranking of attended nodes is unconditioned on the query
    node. In contrast, in GATv2, every node can attend to any other node.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 bias: bool = True, share_weights: bool = False, **kwargs):
        super(GATConv, self).__init__(node_dim=0, **kwargs)

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
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = nn.Linear(in_channels, heads * out_channels,
                                       bias=bias)
        else:
            self.lin_l = nn.Linear(in_channels[0], heads * out_channels, True)
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = nn.Linear(in_channels[1], heads * out_channels, True)

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
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x_l: OptTensor = x
            x_r: OptTensor = x
        else:
            x_l, x_r = x[0], x[1]

        x_l = self.lin_l(x_l).view(-1, H, C)
        if self.share_weights:
            x_r = self.lin_l(x_r).view(-1, H, C)
        else:
            x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if return_attention_weights:
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        if edge_attr is not None:
            assert self.lin_edge is not None
            edge_emb = self.lin_edge(edge_attr).view(-1, self.heads,
                                                     self.out_channels)
            x += edge_emb
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class RelationalGAT(nn.Module):
    def __init__(self, node_feature_dim: int, num_relations: int, num_layers: int = 2,
                 dropout_prob: float = 0.3, heads: int = 4, relation_embedding_dim: int = 64):
        super(RelationalGAT, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, relation_embedding_dim)
        self.layers = nn.ModuleList()
        hidden_channels = node_feature_dim // heads
        for index in range(num_layers):
            is_last_layer = (index == num_layers - 1)
            out_channels = node_feature_dim if is_last_layer else hidden_channels
            heads = 1 if is_last_layer else heads
            concat = False if is_last_layer else True
            self.layers.append(
                GATConv(
                    in_channels=node_feature_dim,
                    out_channels=out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout_prob,
                    edge_dim=relation_embedding_dim
                )
            )
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, node_features, edge_index_list, edge_type_list, num_nodes_list):
        if not edge_index_list or all(e.numel() == 0 for e in edge_index_list):
            return node_features
        edge_index_batch = []
        edge_type_batch = []
        offset = 0
        for i, edge_index in enumerate(edge_index_list):
            edge_index_batch.append(edge_index + offset)
            edge_type_batch.append(edge_type_list[i])
            offset += num_nodes_list[i]
        edge_index_batch = torch.cat(edge_index_batch, dim=1).contiguous()
        edge_type_batch = torch.cat(edge_type_batch, dim=0).contiguous()
        edge_attr = self.relation_embedding(edge_type_batch)
        x = node_features
        attentions = None
        for i, layer in enumerate(self.layers):
            gcn_update, attentions = layer(x, edge_index_batch, edge_attr=edge_attr, return_attention_weights=True)
            x = x + self.dropout(self.activation(gcn_update))
        edge_index_with_self_loops, attention_weights_full = attentions
        is_not_self_loop = edge_index_with_self_loops[0] != edge_index_with_self_loops[1]
        edges_attention = attention_weights_full[is_not_self_loop]
        edges_index = edge_index_with_self_loops[:, is_not_self_loop]
        return x, (edges_index, edges_attention)
