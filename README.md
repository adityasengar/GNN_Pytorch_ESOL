# GCN for Molecular Solubility (ESOL) Prediction

This project provides a complete, script-based pipeline for training a Graph Convolutional Network (GCN) to predict the water solubility of molecules (ESOL dataset). It uses PyTorch Geometric for the GNN implementation and RDKit for molecule processing.

The original analysis was performed in a Jupyter Notebook and has been refactored into a structured command-line application.

## Project Overview

The pipeline executes the following steps:
1.  **Data Loading:** Automatically downloads the `MoleculeNet ESOL` dataset using PyTorch Geometric.
2.  **Model Training:** Trains a GCN model on the dataset to learn the relationship between molecular structure and solubility.
3.  **Model Saving:** Saves the trained model weights for persistent use.
4.  **Inference:** Provides a script to predict the ESOL value for a new molecule given its SMILES string.

## Project Structure

-   `main.py`: The main CLI script for training the GCN model.
-   `predict.py`: The CLI script for making predictions on new SMILES strings.
-   `src/model.py`: Defines the GCN architecture.
-   `src/train.py`: Contains the training and evaluation loops.
-   `data/`: Directory where the ESOL dataset is downloaded and stored.
-   `models/`: Directory where trained models are saved.
-   `requirements.txt`: Lists all necessary Python dependencies.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/adityasengar/GNN_Pytorch_ESOL.git
    cd GNN_Pytorch_ESOL
    ```

2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies. PyTorch and PyTorch Geometric installation can be specific to your system (CPU/GPU). Please follow the official instructions on their websites. Then, install the packages from the requirements file:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Training the Model

To train the GCN model on the ESOL dataset:

```bash
python main.py --epochs 100 --lr 0.001 --batch_size 64
```

This will:
-   Download the ESOL dataset to the `data/` directory (if not already present).
-   Train the model for 100 epochs.
-   Save the trained model to `models/gcn_esol_model.pth`.

### 2. Making Predictions

To predict the ESOL value for a molecule (e.g., caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`):

```bash
python predict.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --model_path models/gcn_esol_model.pth
```

This will load the pre-trained model and output the predicted solubility value.

### Command-Line Arguments

-   `main.py`:
    -   `--batch_size`: Batch size for training. (Default: 32)
    -   `--epochs`: Number of epochs to train. (Default: 50)
    -   `--lr`: Learning rate. (Default: 0.001)
    -   `--model_dir`: Directory to save the trained model. (Default: `models`)

-   `predict.py`:
    -   `smiles`: The SMILES string of the molecule to predict. (Required)
    -   `--model_path`: Path to the trained model file. (Default: `models/gcn_esol_model.pth`)
    -   `--num_features`: Number of input features the model was trained with. (Default: 75)