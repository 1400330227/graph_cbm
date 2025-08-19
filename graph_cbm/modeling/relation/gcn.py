import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import add_self_loops
except ImportError:
    print("PyTorch Geometric not found. Please install it to use the GCN module.")
    GCNConv = None
    add_self_loops = None


def unpad_sequence(x, num_objs):
    recovered_chunks = []
    for i in range(len(num_objs)):
        seq_len = num_objs[i]
        recovered_chunks.append(x[i, :seq_len, :])
    recovered_x = torch.cat(recovered_chunks, dim=0)
    return recovered_x


class SpatialGCN(nn.Module):
    def __init__(self, node_feature_dim: int, num_layers: int = 2, dropout_prob: float = 0.3):
        super(SpatialGCN, self).__init__()
        if GCNConv is None:
            raise ImportError("GCNConv could not be imported. Please ensure PyTorch Geometric is installed correctly.")
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(node_feature_dim, node_feature_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, node_features: torch.Tensor, edge_index_list: list, num_nodes_list: list) -> torch.Tensor:
        flat_node_features = unpad_sequence(node_features, num_nodes_list)
        if not edge_index_list or all(e.numel() == 0 for e in edge_index_list):
            return node_features
        edge_index_batch = []
        offset = 0
        for i, edge_index in enumerate(edge_index_list):
            edge_index_batch.append(edge_index + offset)
            offset += num_nodes_list[i]
        edge_index_batch = torch.cat(edge_index_batch, dim=1).contiguous()
        edge_index_with_self_loops, _ = add_self_loops(edge_index_batch, num_nodes=node_features.size(0))
        x = flat_node_features
        for layer in self.layers:
            gcn_update = layer(x, edge_index_with_self_loops)
            x = x + self.dropout(self.activation(gcn_update))
        refined_flat_features = x
        refined_features_list = refined_flat_features.split(num_nodes_list, dim=0)
        refined_batched_sequence = pad_sequence(refined_features_list, batch_first=True, padding_value=0.0)
        return refined_batched_sequence
        # return x


if __name__ == '__main__':
    # 1. 定义模型参数
    FEATURE_DIM = 64
    NUM_GCN_LAYERS = 2
    BATCH_SIZE = 2

    # 2. 准备模拟的批处理数据
    num_nodes_g1 = 3
    node_features_g1 = torch.randn(num_nodes_g1, FEATURE_DIM)
    edge_index_g1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    num_nodes_g2 = 4
    node_features_g2 = torch.randn(num_nodes_g2, FEATURE_DIM)
    edge_index_g2 = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long)

    # 3. 将数据整合成新的 `forward` 方法所要求的格式
    # a. 将每个图的节点特征放入一个列表，然后用 pad_sequence 打包
    features_list = [node_features_g1, node_features_g2]
    # 形状: (2, 4, 64)，其中图1的最后一个节点是填充的
    batched_sequence_input = pad_sequence(features_list, batch_first=True)

    # b. 边索引列表 (保持不变)
    all_edge_indices = [edge_index_g1, edge_index_g2]

    # c. 节点数量列表 (保持不变)
    all_num_nodes = [num_nodes_g1, num_nodes_g2]

    print("--- 输入数据 ---")
    print(f"输入序列张量形状: {batched_sequence_input.shape}")
    print(f"节点数量列表: {all_num_nodes}")
    print("-" * 20)

    # 4. 实例化并调用模型
    spatial_gcn_model = SpatialGCN(node_feature_dim=FEATURE_DIM, num_layers=NUM_GCN_LAYERS)

    refined_sequence_output = spatial_gcn_model(
        node_features=batched_sequence_input,
        edge_index_list=all_edge_indices,
        num_nodes_list=all_num_nodes
    )

    # 5. 查看输出结果
    print("\n--- 输出结果 ---")
    print(f"输出序列张量形状: {refined_sequence_output.shape}")
    print("-" * 20)

    # 验证：输出的序列张量形状应该和输入的完全一样
    assert batched_sequence_input.shape == refined_sequence_output.shape
    print("成功！输出的序列形状与输入一致。")
