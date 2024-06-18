import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_3d_mol(smiles):
    """Converts a SMILES string to an RDKit molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def get_atom_features(atom):
    """Gets a one-hot encoded feature vector for an atom."""
    atomic_num = atom.GetAtomicNum()
    # Simple one-hot encoding for a few common atoms
    features = torch.zeros(5)
    if atomic_num == 6:  # C
        features[0] = 1
    elif atomic_num == 8:  # O
        features[1] = 1
    elif atomic_num == 7:  # N
        features[2] = 1
    elif atomic_num == 16: # S
        features[3] = 1
    else:  # Other
        features[4] = 1
    return features

def mol_to_graph_data(mol):
    """Converts an RDKit molecule to a PyG Data object for e3nn."""
    if mol is None:
        return None

    # Get atom features and positions
    atom_features = []
    positions = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])

    x = torch.stack(atom_features, dim=0)
    pos = torch.tensor(positions, dtype=torch.float)

    # Get edge index for connectivity
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
    
    if not edge_index: # Handle single-atom molecules
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, pos=pos)

def esol_pre_transform(data):
    """
    A pre_transform function for the ESOL dataset.
    It generates 3D coordinates and replaces the original features.
    """
    mol = smiles_to_3d_mol(data.smiles)
    if mol is None:
        # If RDKit fails, return a graph with a single NaN node
        # This will be filtered out later.
        return Data(x=torch.tensor([[float('nan')]]), edge_index=torch.empty((2,0), dtype=torch.long), pos=torch.empty((0,3)))

    graph = mol_to_graph_data(mol)
    graph.y = data.y
    return graph
