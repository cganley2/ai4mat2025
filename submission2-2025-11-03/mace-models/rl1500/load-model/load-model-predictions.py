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
from pathlib import Path

import jax

train_set_path = '/scratch/cganley2/rl1500/rl-train1500.extxyz'
valid_set_path = '/scratch/cganley2/rl1500/rl-valid1500.extxyz'
test_set_path = '/scratch/cganley2/rl1500/rl-test13500.extxyz'
epochs = 100
graph_cutoff_angstrom = 5.0

reader = ExtxyzReader(
    ExtxyzReader.Config(
        train_dataset_paths=train_set_path,
        valid_dataset_paths=valid_set_path,
        test_dataset_paths=test_set_path,
    )
)


builder_config = GraphDatasetBuilder.Config(
    graph_cutoff_angstrom=graph_cutoff_angstrom,
    batch_size=25,
)

builder = GraphDatasetBuilder(reader, builder_config)
builder.prepare_datasets() # This step is required to compute all dataset information (used later on by most MLIP model)

train_set, validation_set, test_set = builder.get_splits()

mlip_network = Mace(
    Mace.Config(num_channels=128, correlation=2),
    builder.dataset_info,
)

force_field = ForceField.from_mlip_network(mlip_network)

# Find out what is the most recent epoch that was saved
checkpoints = os.listdir("/scratch/cganley2/rl1500/training/model_training/model")
max_epoch_num = max(int(num) for num in checkpoints) - 1

# Load the parameters from the checkpoint
loaded_params_via_ckpt = load_parameters_from_checkpoint(
    local_checkpoint_dir=Path("/scratch/cganley2/rl1500/training/model_training/model").resolve(),
    initial_params=force_field.params,
    epoch_to_load=max_epoch_num,
    load_ema_params=False,
)

# Create a new force field with those parameters
loaded_force_field = ForceField(force_field.predictor, loaded_params_via_ckpt)

test_structures = ase_read(test_set_path, index=':')
predictions = run_batched_inference(test_structures, loaded_force_field, batch_size=50)

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
    
plt.rcParams.update({'font.size': 14})

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
