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
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Acknowledgments
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- MoleculeNet: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.MoleculeNet
