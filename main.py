from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
import torch
from src.model import GCN
from src.train import train_epoch, evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MoleculeNet(root='./data', name='ESOL')
    print("Dataset loaded successfully.")

    train_dataset = dataset[:800]
    test_dataset = dataset[800:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = GCN(dataset.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_rmse = evaluate(model, test_loader, device)
        print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test RMSE: {test_rmse:.4f}')
    print("Training finished.")

if __name__ == '__main__':
    main()