import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class Graph(torch.nn.Module):
    def __init__(
            self,
            num_nodes,
            num_relations,
            input_dim=1280,
            hidden_dim=512,
            out_dim=256,
            num_layers=2,
            dropout=0.5,
            num_bases=None
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, input_dim)
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(input_dim, hidden_dim, num_relations, num_bases))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases))
        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations, num_bases))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x


if __name__ == '__main__':
    num_nodes = 8
    num_relations = 10
    input_dim = 256
    hidden_dim = 512
    out_dim = 256

    edge_index = torch.randint(0, num_nodes, [2, num_nodes])
    edge_type = torch.randint(0, num_relations, [num_nodes])

    rgcn = Graph(num_nodes, input_dim, hidden_dim, out_dim, num_relations, num_layers=2, dropout=0.5)
    x = rgcn(edge_index, edge_type)
    print(x.shape)
