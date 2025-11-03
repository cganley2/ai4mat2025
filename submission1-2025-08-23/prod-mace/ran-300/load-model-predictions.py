# For dataset loading
from mlip.data import GraphDatasetBuilder, ExtxyzReader

# For model
from mlip.models import Mace, Nequip, Visnet, ForceField

# For optimizer
from mlip.training import get_default_mlip_optimizer, OptimizerConfig

# For loss function
from mlip.models.loss import MSELoss

# For training
from mlip.training import TrainingLoop
from mlip.models.model_io import save_model_to_zip, load_model_from_zip
from mlip.models.params_loading import load_parameters_from_checkpoint
from mlip.inference import run_batched_inference

# For logging
from mlip.training import TrainingIOHandler, log_metrics_to_line
from mlip.training.training_io_handler import LogCategory

# Other
import logging
import os
import matplotlib.pyplot as plt
from ase.io import read as ase_read
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

import jax

# train_set_path = '/scratch/cganley2/mlip/Maximize_Diversity_using_RL_Dataset/train-set-300.extxyz'
# valid_set_path = '/scratch/cganley2/mlip/Maximize_Diversity_using_RL_Dataset/validation-set-1200.extxyz'
test_set_path = '/scratch/cganley2/mlip/Maximize_Diversity_using_RL_Dataset/test-set-11703.extxyz'

optimized_force_field = load_model_from_zip(Mace, load_path='/home/cganley2/ai4mat2025/mlip/submission1-2025-08-23/prod-mace/ran-300/final_model.zip')

test_structures = ase_read(test_set_path, index=':')
predictions = run_batched_inference(test_structures, optimized_force_field, batch_size=50)

def rmse(true_vals, pred_vals):
    return np.sqrt(np.mean((true_vals - pred_vals) ** 2))

parity_df = pd.DataFrame({'DFT Energy (eV)': [test_structures[i].get_potential_energy() for i in range(len(test_structures))],
                          'MACE Energy Prediction (eV)': [predictions[i].energy for i in range(len(predictions))],
                          'Per-Frame Chemical Symbols': [test_structures[i].get_chemical_symbols() for i in range(len(test_structures))],
                          'DFT Forces (eV/Ang.)': [test_structures[i].get_forces() for i in range(len(test_structures))],
                          'MACE Forces Prediction (eV/Ang.)': [predictions[i].forces for i in range(len(predictions))],
                          })


# Plotting helper
def parity_plot(ax, x, y, title):
    ax.scatter(x, y, alpha=0.6, edgecolors='w', linewidth=0.5)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y = x')
    ax.set_xlabel('DFT')
    ax.set_ylabel('MACE Prediction')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    # Compute and display RMSE
    error = rmse(x, y)
    r2 = r2_score(x, y)
    ax.text(0.05, 0.95, f'RMSE = {error:.4f}\n$R^2$ = {r2:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))

energy_parity_figure, energy_parity_axes = plt.subplots(figsize=(8,8))
parity_plot(energy_parity_axes, parity_df['DFT Energy (eV)'], parity_df['MACE Energy Prediction (eV)'], 'Energy Parity (eV)')

with open('energy-parity-plot-11703.pkl', 'wb') as f:
   pickle.dump((energy_parity_figure, energy_parity_axes), f)

energy_parity_figure.savefig('energy-parity-plot-11703.png', dpi=600)

# Flatten all atom forces from both columns into 1D arrays for a component, e.g. x, or overall magnitude
def flatten_forces(df, col, component=None):
    values = []
    for forces_per_atom in df[col]:
        for force_vec in forces_per_atom:
            if component is not None:
                values.append(force_vec[component])  # x=0, y=1, z=2
            else:
                # Use norm if no specific component given
                values.append(np.linalg.norm(force_vec))
    return np.array(values)

x_dft = flatten_forces(parity_df, 'DFT Forces (eV/Ang.)', component = 0)
y_dft = flatten_forces(parity_df, 'DFT Forces (eV/Ang.)', component = 1)
z_dft = flatten_forces(parity_df, 'DFT Forces (eV/Ang.)', component = 2)
force_norm_dft = flatten_forces(parity_df, 'DFT Forces (eV/Ang.)', component = None)

x_model = flatten_forces(parity_df, 'MACE Forces Prediction (eV/Ang.)', component = 0)
y_model = flatten_forces(parity_df, 'MACE Forces Prediction (eV/Ang.)', component = 1)
z_model = flatten_forces(parity_df, 'MACE Forces Prediction (eV/Ang.)', component = 2)
force_norm_model = flatten_forces(parity_df, 'MACE Forces Prediction (eV/Ang.)', component = None)

# Setup 2x2 scatter plot grid
force_parity_figure, force_parity_axes = plt.subplots(2, 2, figsize=(12, 12))

# Create each parity plot
parity_plot(force_parity_axes[0, 0], force_norm_dft, force_norm_model, 'Force Norm Parity (eV/Ang.)')
parity_plot(force_parity_axes[0, 1], x_dft, x_model, 'Force X-component Parity (eV/Ang.)')
parity_plot(force_parity_axes[1, 0], y_dft, y_model, 'Force Y-component Parity (eV/Ang.)')
parity_plot(force_parity_axes[1, 1], z_dft, z_model, 'Force Z-component Parity (eV/Ang.)')

force_parity_figure.tight_layout()

with open('force-parity-plots-11703.pkl', 'wb') as f:
   pickle.dump((force_parity_figure, force_parity_axes), f)

force_parity_figure.savefig('force-parity-plots-11703.png', dpi=600)
plt.clf()