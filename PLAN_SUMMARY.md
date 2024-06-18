# GNN_Pytorch_ESOL - Remaining Development Plan

This document summarizes the current state of the `GNN_Pytorch_ESOL` project and outlines the next steps, including the planned commits and their objectives. This plan follows the timeline from February 2024 to June 2024.

## Current State

As of the last commit (March 20, 2024), the project has been refactored from a Jupyter Notebook into a modular Python application:

-   **Project Structure:**
    -   `main.py`: The main execution script with a Command-Line Interface (CLI) for training.
    -   `src/model.py`: Defines the Graph Convolutional Network (GCN) architecture.
    -   `src/train.py`: Contains the training and evaluation loops.
    -   `data/`: Directory for the MoleculeNet ESOL dataset.
    -   `.gitignore`: Configured for Python projects and specific data/model artifacts.

-   **Functionality Implemented:**
    -   Loading of the ESOL dataset.
    -   GCN model definition.
    -   Training and evaluation (RMSE).
    -   CLI for setting hyperparameters (batch size, epochs, learning rate).

## Remaining Tasks (Planned Commits)

Here are the next steps to complete the project as planned:

### **Commit 4 (Apr 15, 2024): `feat: Implement model saving and inference script`**

-   **Objective:** Introduce model persistence. After training, the best performing model should be saved to disk. A separate script or an extension of `main.py` will allow loading this model to make predictions on new, unseen molecular data (e.g., provided as SMILES strings).
-   **Actions:**
    -   Modify `src/train.py` to save the trained GCN model (and potentially the dataset's `num_features` if needed for loading) after training.
    -   Create a new script, `predict.py`, that:
        -   Takes a SMILES string or a file of SMILES strings as input.
        -   Loads the pre-trained GCN model.
        -   Converts new SMILES into `torch_geometric.data.Data` objects.
        -   Performs inference to predict ESOL values.

### **Commit 5 (June 18, 2024): `docs: Add requirements.txt and final README`**

-   **Objective:** Professionalize the project with comprehensive documentation and dependency management.
-   **Actions:**
    -   Create a `requirements.txt` file listing all necessary Python libraries (`torch`, `torch-geometric`, `rdkit-pypi`, `scikit-learn`, `matplotlib`, `numpy`).
    -   Update the existing `README.md` to reflect the new script-based structure, explain how to train the model, how to use the new prediction script, and detail the CLI arguments.

---