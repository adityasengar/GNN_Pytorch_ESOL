# Geometric Deep Learning for Molecular Solubility

This library implements a Message Passing Neural Network (MPNN) to predict aqueous solubility (ESOL). Unlike standard QSAR methods, this approach leverages the graph topology of molecules, utilizing both atom and bond-level features.

## Key Features

- **Graph Featurization**: Custom RDKit extraction of atom and bond attributes.
- **Calibrated Uncertainty**: Implements Aleatoric Uncertainty quantification using Heteroscedastic Loss (NLL), allowing the model to flag low-confidence predictions on novel scaffolds.
- **Production Ready**: Packaged as a standard Python library for easy integration into drug discovery workflows.

## Installation

```bash
git clone https://github.com/adityasengar/GNN_Pytorch_ESOL.git
cd GNN_Pytorch_ESOL
pip install .
```

## Usage

To predict the solubility of a molecule from its SMILES string:

```python
from admet_gnn import ADMETPredictor

# Load the trained model
predictor = ADMETPredictor(model_path='path/to/your/trained_model.pt')

# Predict solubility for a list of SMILES strings
smiles = ['CCO', 'c1ccccc1']
results = predictor.predict(smiles)

for smi, result in zip(smiles, results):
    print(f'SMILES: {smi}')
    print(f'  Predicted Solubility: {result["solubility"]:.2f}')
    print(f'  Uncertainty (stdev): {result["uncertainty"]:.2f}')
    print(f'  Confidence: {result["confidence"]}')
```

## Training

To train the model, you need a CSV file with a 'smiles' column and a 'solubility' column.

```bash
python -m src.train --data_path /path/to/your/data.csv
```