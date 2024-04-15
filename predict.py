import torch
import argparse
import os
from rdkit import Chem
from torch_geometric.data import Data

from src.model import GCN

def smiles_to_graph(smiles):
    """Converts a SMILES string to a torch_geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    x = torch.tensor(atom_features, dtype=torch.float).unsqueeze(1)

    # Get bond features (edge index)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i]) # Edges are undirected
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Manually create a Data object
    # Note: num_features must match what the model was trained on.
    # This is a simplified feature extraction. A robust implementation would use
    # the same feature extraction function for both training and prediction.
    data = Data(x=x, edge_index=edge_index)
    return data

def main():
    parser = argparse.ArgumentParser(description="Predict ESOL for a given SMILES string.")
    parser.add_argument("smiles", type=str, help="The SMILES string of the molecule to predict.")
    parser.add_argument("--model_path", type=str, default="models/gcn_esol_model.pth", help="Path to the trained model file.")
    # The number of features needs to be known. In a real app, this would be saved with the model.
    parser.add_argument("--num_features", type=int, default=75, help="Number of input features the model was trained with.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}. Please train the model first.")
        return

    # Load model
    print("Loading pre-trained model...")
    model = GCN(num_features=args.num_features).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Prepare data
    print(f"Preparing graph from SMILES: {args.smiles}")
    data = smiles_to_graph(args.smiles)
    if data is None:
        print("Error: Invalid SMILES string.")
        return
        
    # The feature vector size in this simple conversion might not match the model's expected input.
    # We will pad it with zeros as a simple fix.
    if data.x.shape[1] < args.num_features:
        padding = torch.zeros(data.x.shape[0], args.num_features - data.x.shape[1])
        data.x = torch.cat([data.x, padding], dim=1)
    
    data = data.to(device)

    # Predict
    with torch.no_grad():
        prediction = model(data)
    
    print(f"\nPredicted ESOL value: {prediction.item():.4f}")

if __name__ == '__main__':
    main()
