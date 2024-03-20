import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x.float(), edge_index.long())
        x = F.relu(x)
        x = self.conv2(x, edge_index.long())
        x = F.relu(x)
        x = self.conv3(x, edge_index.long())
        x = global_add_pool(x, batch)
        return x.squeeze()
