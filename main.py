import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
import argparse

from src.model import GCN
from src.train import train_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description="GCN for ESOL Prediction")
    parser.add_argument('--batch_size', type=int, default=32, help="Input batch size for training.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MoleculeNet(root='./data', name='ESOL')
    print("Dataset loaded successfully.")

    train_dataset = dataset[:800]
    test_dataset = dataset[800:]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = GCN(dataset.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_rmse = evaluate(model, test_loader, device)
        print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test RMSE: {test_rmse:.4f}')
    print("Training finished.")

if __name__ == '__main__':
    main()
