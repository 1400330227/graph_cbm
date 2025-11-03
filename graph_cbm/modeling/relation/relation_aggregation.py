import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch_geometric.typing import (Adj, PairTensor)
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, scatter
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add, scatter_softmax, scatter_mean


class RelationAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, num_channels=2, concat=True, negative_slope=0.2, dropout=0.0,
                 add_self_loops=False, edge_dim=None, fill_value='mean', bias=False, share_weights=True,
                 use_channel_bias=True, **kwargs):
        super(RelationAggregation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.use_channel_bias = use_channel_bias

        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = self.lin_l if share_weights else nn.Linear(in_channels, out_channels, bias=bias)

        self.channel_bias_l = None
        self.channel_bias_r = None
        if use_channel_bias:
            self.channel_bias_l = Parameter(Tensor(num_channels, out_channels))
            self.channel_bias_r = Parameter(Tensor(num_channels, out_channels)) if not share_weights else None

        self.att_shared = Parameter(Tensor(1, out_channels))
        self.att_scale = Parameter(Tensor(num_channels, 1))

        self.edge_bias = None
        self.lin_edge = None
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
            self.edge_bias = Parameter(Tensor(out_channels, num_channels)) if use_channel_bias else None

        if bias and concat:
            self.bias = Parameter(Tensor(num_channels * out_channels))
        elif bias and not concat:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        if not self.share_weights and self.lin_r is not None:
            glorot(self.lin_r.weight)
        if self.channel_bias_l is not None:
            zeros(self.channel_bias_l)
        if self.channel_bias_r is not None:
            zeros(self.channel_bias_r)
        if self.lin_edge is not None:
            glorot(self.lin_edge.weight)
        if self.edge_bias is not None:
            zeros(self.edge_bias)
        glorot(self.att_shared)
        glorot(self.att_scale)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        num_nodes = x.size(0) if isinstance(x, Tensor) else x[0].size(0)

        x_l_base = self.lin_l(x)
        x_r_base = self.lin_r(x) if not self.share_weights else x_l_base

        x_l_channels = []
        x_r_channels = []
        for k in range(self.num_channels):
            x_l_k = x_l_base
            x_r_k = x_r_base
            if self.channel_bias_l is not None:
                x_l_k = x_l_k + self.channel_bias_l[k].unsqueeze(0)
            if self.channel_bias_r is not None:
                x_r_k = x_r_k + self.channel_bias_r[k].unsqueeze(0)
            x_l_channels.append(x_l_k)
            x_r_channels.append(x_r_k)
        x_l = torch.stack(x_l_channels, dim=-1)
        x_r = torch.stack(x_r_channels, dim=-1)

        if self.add_self_loops:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, self.fill_value, num_nodes)

        source_nodes, target_nodes = edge_index[0], edge_index[1]
        x_i = x_l[target_nodes]  # (E, out_channels, num_channels)
        x_j = x_r[source_nodes]  # (E, out_channels, num_channels)
        attention_input = x_i + x_j

        edge_emb = torch.zeros_like(x_j)
        if self.lin_edge is not None and edge_attr is not None:
            edge_emb = self.lin_edge(edge_attr)  # (E, out_channels)
            edge_emb = edge_emb.unsqueeze(-1)  # (E, out_channels, 1)
            edge_emb = edge_emb.expand(-1, -1, self.num_channels)  # (E, out_channels, num_channels)
            if self.edge_bias is not None:
                edge_emb = edge_emb + self.edge_bias.unsqueeze(0)
            attention_input += edge_emb

        attention_input = F.leaky_relu(attention_input, self.negative_slope)
        # (E, out_channels, num_channels) * (1, out_channels, 1) -> (E, num_channels)
        # alpha = (attention_input * self.att_shared.unsqueeze(-1)).sum(dim=1)
        # alpha = alpha * self.att_scale.t().unsqueeze(0)  # (E, num_channels)

        alpha_channels = []
        for k in range(self.num_channels):
            attention_input_k = attention_input[:, :, k]  # (E, out_channels)
            # (E, out_channels) * (1, out_channels) -> (E, out_channels) -> sum -> (E,)
            alpha_k = (attention_input_k * self.att_shared).sum(dim=-1)
            alpha_k = alpha_k * self.att_scale[k]  # (E,)
            alpha_k = softmax(alpha_k, target_nodes, num_nodes=num_nodes)  # (E,)
            alpha_channels.append(alpha_k)
        alpha = torch.stack(alpha_channels, dim=-1)  # (E, num_channels)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        weighted_messages = (x_j + edge_emb) * alpha.unsqueeze(1)  # (E, out_channels, num_channels)

        out_channels = []
        for k in range(self.num_channels):
            out_k = scatter(weighted_messages[:, :, k], target_nodes, dim=0, dim_size=num_nodes, reduce='sum')
            out_channels.append(out_k)

        if self.concat:
            out = torch.cat(out_channels, dim=-1)  # (V, num_channels * out_channels)
        else:
            out = torch.stack(out_channels, dim=-1).mean(dim=-1)  # out: (V, out_channels)

        if self.bias is not None:
            out += self.bias
        # self.concat = True，(V, num_channels * out_channels)
        # self.concat = False，(V, out_channels)
        return out, (edge_index, alpha)


class ObjectAggregation(nn.Module):
    def __init__(self, representation_dim, num_channels=2, use_channel_bias=True):
        super(ObjectAggregation, self).__init__()
        self.att = Parameter(torch.Tensor(1, representation_dim))
        self.num_channels = num_channels
        self.use_channel_bias = use_channel_bias

        self.att_shared = Parameter(torch.Tensor(1, representation_dim))
        self.att_scale = Parameter(torch.Tensor(num_channels, 1))
        self.channel_bias = Parameter(Tensor(num_channels, representation_dim)) if use_channel_bias else None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_shared)
        glorot(self.att_scale)
        if self.channel_bias is not None:
            zeros(self.channel_bias)

    def forward(self, x_aggregation, num_objs):
        batch_size = len(num_objs)
        batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=x_aggregation.device),
            torch.tensor(num_objs, device=x_aggregation.device)
        )
        # (batch_size, representation_dim) or (batch_size, representation_dim, num_channels)
        scene_context_features = scatter_mean(x_aggregation, batch_idx, dim=0)
        # (total_nodes, representation_dim) or (total_nodes, representation_dim, num_channels)
        attention_input = x_aggregation + scene_context_features[batch_idx]
        attention_input = F.leaky_relu(attention_input, negative_slope=0.2)

        # 为每个通道计算注意力logits
        attn_logits_channels = []
        for k in range(self.num_channels):
            if attention_input.dim() == 3:  #(V, representation_dim, num_channels)
                attention_input_k = attention_input[:, :, k]  # (V, representation_dim)
            else:
                attention_input_k = attention_input  # (V, representation_dim)
            if self.channel_bias is not None:
                # (V, representation_dim)
                attention_input_k = attention_input_k + self.channel_bias[k].unsqueeze(0)
            attn_logits_k = (attention_input_k * self.att_shared).sum(dim=-1)  # (total_nodes,)
            attn_logits_k = attn_logits_k * self.att_scale[k]
            attn_logits_channels.append(attn_logits_k)
        attn_logits = torch.stack(attn_logits_channels, dim=-1)  # attn_logits: (V, num_channels)

        attn_weights_channels = []
        for k in range(self.num_channels):
            attn_weights_k = scatter_softmax(attn_logits[:, k], batch_idx, dim=0)  # (V,)
            attn_weights_channels.append(attn_weights_k)
        attn_weights = torch.stack(attn_weights_channels, dim=-1)  # attn_weights: (V, num_channels)

        scene_features_channels = []
        for k in range(self.num_channels):
            if x_aggregation.dim() == 3:
                # (V, representation_dim)
                weighted_features = x_aggregation[:, :, k] * attn_weights[:, k].unsqueeze(-1)
            else:
                # (V, representation_dim)
                weighted_features = x_aggregation * attn_weights[:, k].unsqueeze(-1)
            scene_features_k = scatter_add(weighted_features, batch_idx, dim=0)  # (batch_size, representation_dim)
            scene_features_channels.append(scene_features_k)
        # scene_features: (batch_size, representation_dim, num_channels)
        scene_features = torch.stack(scene_features_channels, dim=-1)

        return scene_features, attn_weights


class Aggregation(nn.Module):
    def __init__(self, node_feature_dim, num_relations, num_layers=2, dropout_prob=0.3, num_channels=4,
                 relation_dim=64, hidden_dim=64, concat=True, share_weights=True, use_channel_bias=True):
        super(Aggregation, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, relation_dim)
        self.relation_aggregators = nn.ModuleList()
        self.object_aggregators = nn.ModuleList()
        self.num_layers = num_layers
        relation_output_dim = num_channels * hidden_dim if concat else hidden_dim

        for index in range(num_layers):
            self.relation_aggregators.append(
                RelationAggregation(
                    in_channels=node_feature_dim,
                    out_channels=hidden_dim,
                    num_channels=num_channels,
                    concat=concat,
                    dropout=dropout_prob,
                    edge_dim=relation_dim,
                    share_weights=share_weights,
                    use_channel_bias=use_channel_bias
                )
            )
            self.add_module(f'relation_proj_{index}', nn.Linear(relation_output_dim, node_feature_dim))
            self.object_aggregators.append(
                ObjectAggregation(
                    representation_dim=node_feature_dim,
                    num_channels=num_channels,
                    use_channel_bias=use_channel_bias)
            )
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, node_features, edge_index_list, edge_type_list, num_nodes_list):
        if not edge_index_list or all(e.numel() == 0 for e in edge_index_list):
            x = node_features
            scene_features, object_attentions = self.object_aggregator(x, num_nodes_list)
            return scene_features, x, (object_attentions, None, None)
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
        relation_attentions = []
        relation_indexes = []
        object_attentions = []
        scene_features = []
        for i, (rel_layer, obj_layer) in enumerate(zip(self.relation_aggregators, self.object_aggregators)):
            x_update, attentions = rel_layer(x, edge_index_batch, edge_attr=edge_attr)
            projection_layer = getattr(self, f'relation_proj_{i}')
            x_update_projected = projection_layer(x_update)  # [V, node_feature_dim]
            x = x + self.dropout(self.activation(x_update_projected))
            scene_feature, obj_attention = obj_layer(x, num_nodes_list)

            attention_index, attention_weight = attentions
            relation_indexes = attention_index
            relation_attentions.append(attention_weight)
            object_attentions.append(obj_attention)
            scene_features.append(scene_feature)
        relation_attentions = torch.stack(relation_attentions, dim=0)
        object_attentions = torch.stack(object_attentions, dim=0)
        scene_features = torch.stack(scene_features, dim=0).mean(dim=0)

        # for i, layer in enumerate(self.relation_aggregators):
        #     x_update, attentions = layer(x, edge_index_batch, edge_attr=edge_attr)
        #     x = x + self.dropout(self.activation(x_update))
        #     attention_index, attention_weight = attentions
        #     relation_indexes = attention_index
        #     relation_attentions.append(attention_weight)
        # relation_attentions = torch.stack(relation_attentions, dim=0)  # (L, V, num_channels)
        #
        # object_attentions = []
        # scene_features = []
        # for i, layer in enumerate(self.object_aggregators):
        #     scene_feature, attentions = layer(x, num_nodes_list)
        #     scene_feature = scene_feature + self.dropout(self.activation(scene_feature))
        #     scene_features.append(scene_feature)
        #     object_attentions.append(attentions)
        # scene_features = torch.stack(scene_features, dim=0).mean(dim=0)
        # object_attentions = torch.stack(object_attentions, dim=0)  # (L, V, num_channels)

        return scene_features, (object_attentions, relation_attentions, relation_indexes)
