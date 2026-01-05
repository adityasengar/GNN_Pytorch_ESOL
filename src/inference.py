import torch
from admet_gnn.src.featurizer import MoleculeFeaturizer
from admet_gnn.src.model import UncertaintyGNN
from torch_geometric.data import DataLoader

class ADMETPredictor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = UncertaintyGNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.featurizer = MoleculeFeaturizer()

    def predict(self, smiles_list):
        # 1. Convert SMILES to Graphs
        graphs = [self.featurizer.smiles_to_graph(s) for s in smiles_list]
        loader = DataLoader(graphs, batch_size=len(smiles_list))
        
        # 2. Batch and Predict
        results = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                mu, log_var = self.model(data)
                
                # Convert log_var to standard deviation
                sigma = torch.exp(0.5 * log_var)

                for i in range(len(mu)):
                    results.append({
                        "solubility": mu[i].item(),
                        "uncertainty": sigma[i].item(),
                        "confidence": "LOW" if sigma[i].item() > 0.5 else "HIGH"
                    })
        return results
