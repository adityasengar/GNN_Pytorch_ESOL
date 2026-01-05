import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from admet_gnn.src.featurizer import MoleculeFeaturizer

class ADMETDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None):
        self.filename = filename
        self.featurizer = MoleculeFeaturizer()
        super(ADMETDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_paths))]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, row in self.data.iterrows():
            smiles = row['smiles']
            solubility = row['solubility']
            graph = self.featurizer.smiles_to_graph(smiles)
            graph.y = torch.tensor([[solubility]], dtype=torch.float)
            torch.save(graph, f'{self.processed_dir}/data_{index}.pt')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(f'{self.processed_dir}/data_{idx}.pt')
        return data
