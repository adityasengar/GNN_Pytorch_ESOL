import torch
from torch_geometric.data import DataLoader
from admet_gnn.src.model import UncertaintyGNN
from admet_gnn.src.dataset import ADMETDataset
import argparse

def heteroscedastic_loss(true, mean, log_var):
    """
    Calculates the Negative Log Likelihood loss.
    true: ground truth values
    mean: predicted mean (mu)
    log_var: predicted log variance (s)
    """
    precision = torch.exp(-log_var)
    return torch.mean(0.5 * precision * (true - mean)**2 + 0.5 * log_var)

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        mean, log_var = model(data)
        loss = heteroscedastic_loss(data.y, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = ADMETDataset(root='data', filename=args.data_path)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = UncertaintyGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pt')

if __name__ == '__main__':
    main()
