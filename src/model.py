import torch
from e3nn import o3
from e3nn.nn import Gate, BatchNorm, TensorProduct
from torch_geometric.nn import global_mean_pool

class E3NN_GCN(torch.nn.Module):
    def __init__(self, node_features_irreps, hidden_irreps, output_irreps):
        super().__init__()
        
        # Define the input and output representations for the tensor products
        self.node_features_irreps = o3.Irreps(f"{node_features_irreps}x0e") # Input scalars
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_irreps = o3.Irreps(output_irreps)
        
        # First layer: Lift scalar features to spherical harmonic features
        self.tp1 = TensorProduct(
            self.node_features_irreps,
            "1o",  # Spherical harmonics of degree 1 (from positions)
            self.hidden_irreps,
            shared_weights=False
        )
        
        # Second layer: Interact features
        self.tp2 = TensorProduct(
            self.hidden_irreps,
            "1o",
            self.hidden_irreps,
            shared_weights=False
        )
        
        # Gated non-linearity
        self.gate = Gate(
            self.hidden_irreps, [torch.nn.functional.silu], #Scalars
            self.hidden_irreps, [torch.sigmoid] # Gates
        )
        
        # Final layer: Project to scalar output
        self.output_layer = torch.nn.Linear(self.hidden_irreps.dim, self.output_irreps.dim)

    def forward(self, data):
        x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos

        # Edge vectors
        edge_src, edge_dst = edge_index
        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_sh = o3.spherical_harmonics("1o", edge_vec, normalize=True, normalization='component')

        # First message passing layer
        x_src, x_dst = x[edge_src], x[edge_dst]
        edge_features = self.tp1(x_src, edge_sh)
        x = torch.zeros(data.num_nodes, self.hidden_irreps.dim, device=x.device).scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(edge_features), edge_features)
        
        x = self.gate(x)

        # Second message passing layer
        x_src, x_dst = x[edge_src], x[edge_dst]
        edge_features = self.tp2(x_src, edge_sh)
        x = torch.zeros(data.num_nodes, self.hidden_irreps.dim, device=x.device).scatter_add_(0, edge_dst.unsqueeze(-1).expand_as(edge_features), edge_features)

        x = self.gate(x)
        
        # Global pooling and final prediction
        x = global_mean_pool(x, batch)
        x = self.output_layer(x)
        
        return x.squeeze()