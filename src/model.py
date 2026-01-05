import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

class UncertaintyGNN(torch.nn.Module):
    def __init__(self, num_features=3):
        super(UncertaintyGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)
        
        # Branch for mean prediction
        self.mean_conv = GCNConv(32, 1)

        # Branch for variance prediction
        self.var_conv = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Mean prediction
        mean_x = self.mean_conv(x, edge_index)
        mean_x = global_add_pool(mean_x, batch)

        # Log variance prediction
        log_var_x = self.var_conv(x, edge_index)
        log_var_x = global_add_pool(log_var_x, batch)

        return mean_x.squeeze(), log_var_x.squeeze()
