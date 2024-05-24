import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
import argparse
import os

from src.model import GCN
from src.train import train_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description="GCN for ESOL Prediction")
    parser.add_argument('--batch_size', type=int, default=32, help="Input batch size for training.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train.")
    parser.add_argument('--lr', type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument('--model_dir', type=str, default='models', help="Directory to save the trained model.")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "gcn_esol_model.pth")

    dataset = MoleculeNet(root='./data', name='ESOL')
    print("Dataset loaded successfully.")
    print(f"  - Number of graphs: {len(dataset)}")

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
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()