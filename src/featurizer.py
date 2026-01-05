import torch
from rdkit import Chem
from torch_geometric.data import Data

class MoleculeFeaturizer:
    def _get_atom_features(self, atom):
        # Atomic number
        atomic_num = atom.GetAtomicNum()
        # Hybridization
        hybridization = atom.GetHybridization()
        # Aromaticity
        is_aromatic = atom.GetIsAromatic()
        
        features = [
            atomic_num,
            hybridization,
            is_aromatic
        ]
        return features

    def _get_bond_features(self, bond):
        # Bond type
        bond_type = bond.GetBondTypeAsDouble()
        # Conjugation
        is_conjugated = bond.GetIsConjugated()
        
        features = [
            bond_type,
            is_conjugated
        ]
        return features

    def smiles_to_graph(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        
        # Get atom features
        atom_features = [self._get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Get bond features and edge index
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_features = self._get_bond_features(bond)
            
            edge_indices.append((i, j))
            edge_attrs.append(edge_features)
            
            edge_indices.append((j, i))
            edge_attrs.append(edge_features)
            
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
