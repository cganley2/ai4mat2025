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

train_set_path = '/home/cganley2/mlip/aimd-trajectories/trainingXXX-YYY.extxyz'
valid_set_path = '/home/cganley2/mlip/aimd-trajectories/validation-set-1200.extxyz'
test_set_path = '/home/cganley2/mlip/aimd-trajectories/test-set-1200.extxyz'
epochs = 60
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


# We override some of the default hyperparameters 
# of the model to make it smaller such that this training example becomes more minimal
mlip_network = Mace(
    Mace.Config(num_channels=128, correlation=2),
    builder.dataset_info,
)

# mlip_network = Nequip(
#     Nequip.Config(
#         node_irreps="4x0e + 4x0o + 4x1o + 4x1e + 4x2e + 4x2o",
#         num_layers=2,
#     ),
#     builder.dataset_info,
# )

# mlip_network = Visnet(
#     Visnet.Config(num_channels=16, num_layers=2),
#     builder.dataset_info,
# )

force_field = ForceField.from_mlip_network(mlip_network)

optimizer = get_default_mlip_optimizer()

loss = MSELoss()

training_config = TrainingLoop.Config(num_epochs=epochs)

os.makedirs('training/model_training', exist_ok=True)

io_handler = TrainingIOHandler(
    TrainingIOHandler.Config(
        local_model_output_dir="training/model_training"
    )
)

# The following logger is also attached in the default I/O handler
# that was used in the training above
io_handler.attach_logger(log_metrics_to_line)

# Define a custom logging function that keeps track of validation loss
validation_losses = []
training_losses = []
def _custom_logger(category, to_log, epoch_number):
  if category == LogCategory.EVAL_METRICS:
    validation_losses.append(to_log["loss"])
  if category == LogCategory.TRAIN_METRICS:
    training_losses.append(to_log["loss"])

# Attach our custom logging function to the I/O handler
io_handler.attach_logger(_custom_logger)

training_loop = TrainingLoop(
    train_dataset=train_set,
    validation_dataset=validation_set,
    force_field=force_field,
    loss=loss,
    optimizer=optimizer,
    config=training_config,
    io_handler=io_handler,
)

training_loop.run()

training_loop.test(test_set)

epoch_nums = list(range(len(validation_losses)))

validation_figure, validation_ax = plt.subplots()
validation_ax.plot(epoch_nums, validation_losses)
validation_ax.set_xlabel("Epoch")
validation_ax.set_ylabel("Validation loss")
# plt.xticks(epoch_nums)

with open('validation-losses.pkl', 'wb') as f:
   pickle.dump((validation_figure, validation_ax), f)

validation_figure.savefig('validation-losses.png', dpi=300)
plt.clf()

training_figure, training_ax = plt.subplots()
training_ax.plot(epoch_nums[1:], training_losses)
training_ax.set_xlabel("Epoch")
training_ax.set_ylabel("Training loss")
# plt.xticks(epoch_nums)

with open('training-losses.pkl', 'wb') as f:
   pickle.dump((training_figure, training_ax), f)

training_figure.savefig('training-losses.png', dpi=300)
plt.clf()

optimized_force_field = training_loop.best_model

test_structures = ase_read(test_set_path, index=':')
predictions = run_batched_inference(test_structures, optimized_force_field, batch_size=50)

save_model_to_zip("final_model.zip", optimized_force_field)

def rmse(true_vals, pred_vals):
    return np.sqrt(np.mean((true_vals - pred_vals) ** 2))

parity_df = pd.DataFrame({'DFT Energy (eV)': [test_structures[i].get_potential_energy() for i in range(len(test_structures))],
                          'MACE Energy Prediction (eV)': [predictions[i].energy for i in range(len(predictions))],
                          'Per-Frame Chemical Symbols': [test_structures[i].get_chemical_symbols() for i in range(len(test_structures))],
                          'DFT Forces (eV/Ang.)': [test_structures[i].get_forces() for i in range(len(test_structures))],
                          'MACE Forces Prediction (eV/Ang.)': [predictions[i].forces for i in range(len(predictions))],
                          })

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

# Create each parity plot
parity_plot(force_parity_axes[0, 0], force_norm_dft, force_norm_model, 'Force Norm Parity (eV/Ang.)')
parity_plot(force_parity_axes[0, 1], x_dft, x_model, 'Force X-component Parity (eV/Ang.)')
parity_plot(force_parity_axes[1, 0], y_dft, y_model, 'Force Y-component Parity (eV/Ang.)')
parity_plot(force_parity_axes[1, 1], z_dft, z_model, 'Force Z-component Parity (eV/Ang.)')

force_parity_figure.tight_layout()

with open('force-parity-plots.pkl', 'wb') as f:
   pickle.dump((force_parity_figure, force_parity_axes), f)

force_parity_figure.savefig('force-parity-plots.png', dpi=300)
plt.clf()

energy_parity_figure, energy_parity_axes = plt.subplots(figsize=(8,8))
parity_plot(energy_parity_axes, parity_df['DFT Energy (eV)'], parity_df['MACE Energy Prediction (eV)'], 'Energy Parity (eV)')

with open('energy-parity-plot.pkl', 'wb') as f:
   pickle.dump((energy_parity_figure, energy_parity_axes), f)

energy_parity_figure.savefig('energy-parity-plot.png', dpi=300)

