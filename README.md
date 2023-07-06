# Molecular Property Prediction with Graph Convolutional Networks (GCN)

This repository contains an implementation of Graph Convolutional Networks (GCN) using PyTorch Geometric for predicting molecular properties, specifically focused on water solubility prediction.

## Dataset
The ESOL dataset is used for training and evaluation. It contains a collection of small organic molecules along with their measured water solubility values.

## Model Architecture
The GCN model consists of multiple graph convolutional layers followed by global pooling to generate a fixed-size graph-level representation. The final output layer predicts the water solubility value for each molecule.

## Dependencies
- torch
- torch_geometric
- scikit-learn
- matplotlib
- numpy

## Usage
1. Install the required dependencies: `pip install -r requirements.txt`
2. Run the `train.py` script to train the GCN model.
3. Evaluate the trained model on the test set using the `evaluate.py` script.
4. Use the trained model for making predictions on new molecules using the `predict.py` script.

## Results
The trained GCN model achieves a mean squared error (MSE) of X on the test set, demonstrating its effectiveness in predicting water solubility.

## License
This project is licensed under the [MIT License](LICENSE).

Feel free to use and modify the code according to your needs.

## Acknowledgments
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- MoleculeNet: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.MoleculeNet
