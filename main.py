import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Model Definition ---
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

# --- 2. Training & Evaluation ---
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        data = data.to(device)
        output = model(data)
        y_true.append(data.y.cpu())
        y_pred.append(output.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    return mean_squared_error(y_true, y_pred, squared=False)

# --- 3. Main Execution ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset, it will be downloaded to the specified root directory
    dataset = MoleculeNet(root='./data', name='ESOL')
    print("Dataset loaded successfully.")

    # Split dataset
    train_dataset = dataset[:800]
    test_dataset = dataset[800:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = GCN(dataset.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting training...")
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        test_rmse = evaluate(model, test_loader, device)
        print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test RMSE: {test_rmse:.4f}')
    print("Training finished.")

if __name__ == '__main__':
    main()
