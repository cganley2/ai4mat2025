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

import jax

reader = ExtxyzReader(
    ExtxyzReader.Config(
        train_dataset_paths="/home/cganley2/mlip/aimd-trajectories/training-set-300.extxyz",
        valid_dataset_paths="/home/cganley2/mlip/aimd-trajectories/validation-set-1200.extxyz",
        test_dataset_paths="/home/cganley2/mlip/aimd-trajectories/test-set-1200.extxyz",
    )
)


builder_config = GraphDatasetBuilder.Config(
    graph_cutoff_angstrom=3.0,
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

training_config = TrainingLoop.Config(num_epochs=100)

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
def _custom_logger(category, to_log, epoch_number):
  if category == LogCategory.EVAL_METRICS:
    validation_losses.append(to_log["loss"])

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
plt.plot(epoch_nums, validation_losses)
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.xticks(epoch_nums)
plt.savefig('validation-losses.png', dpi=300)

optimized_force_field = training_loop.best_model

test_structures = ase_read('/home/cganley2/mlip/aimd-trajectories/test-set-1200.extxyz', index=':')
predictions = run_batched_inference(test_structures, optimized_force_field, batch_size=50)

# predictions will be as long as test_structures, with predictions[X].forces and predictions[X].energy attributes
with open('mace-force-predictions-on-test.txt', 'w') as f_forces, open('mace-energy-predictions-on-test.txt', 'w') as f_energy:
  for frame in predictions:
    f_forces.write(f"{frame.forces}\n")
    f_energy.write(f"{frame.energy}\n")

with open('mace-predictions.pkl', 'wb') as f:
  pickle.dump(predictions, f)

save_model_to_zip("final_model.zip", optimized_force_field)

