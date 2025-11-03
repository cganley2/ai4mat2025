import numpy as np
import ase
from ase import io
from sklearn.model_selection import train_test_split

aimd_500k = ase.io.read('/home/cganley2/ai4mat2025/submission2/mace-models/training-data/SiC_3C-aimd-500K.out', index=':')
aimd_1500k = ase.io.read('/home/cganley2/ai4mat2025/submission2/mace-models/training-data/SiC_3C-aimd-1500K.out', index=':')
aimd_2500k = ase.io.read('/home/cganley2/ai4mat2025/submission2/mace-models/training-data/SiC_3C-aimd-2500K.out', index=':')

aimd_500k_indices = np.arange(len(aimd_500k))
aimd_1500k_indices = np.arange(len(aimd_1500k))
aimd_2500k_indices = np.arange(len(aimd_2500k))

training_set_size_per_traj = 300

# select random train/val/test split
train_indices_500k, temp = train_test_split(aimd_500k_indices, test_size=len(aimd_500k_indices)-training_set_size_per_traj, random_state=42)
validation_indices_500k, test_indices_500k = train_test_split(temp, test_size=len(aimd_500k_indices)-(2*training_set_size_per_traj), random_state=42)

train_indices_1500k, temp = train_test_split(aimd_1500k_indices, test_size=len(aimd_1500k_indices)-training_set_size_per_traj, random_state=42)
validation_indices_1500k, test_indices_1500k = train_test_split(temp, test_size=len(aimd_1500k_indices)-(2*training_set_size_per_traj), random_state=42)

train_indices_2500k, temp = train_test_split(aimd_2500k_indices, test_size=len(aimd_2500k_indices)-training_set_size_per_traj, random_state=42)
validation_indices_2500k, test_indices_2500k = train_test_split(temp, test_size=len(aimd_2500k_indices)-(2*training_set_size_per_traj), random_state=42)

train_500k_frames = [aimd_500k[index] for index in train_indices_500k]
validation_500k_frames = [aimd_500k[index] for index in validation_indices_500k]
test_500k_frames = [aimd_500k[index] for index in test_indices_500k]

train_1500k_frames = [aimd_1500k[index] for index in train_indices_1500k]
validation_1500k_frames = [aimd_1500k[index] for index in validation_indices_1500k]
test_1500k_frames = [aimd_1500k[index] for index in test_indices_1500k]

train_2500k_frames = [aimd_2500k[index] for index in train_indices_2500k]
validation_2500k_frames = [aimd_2500k[index] for index in validation_indices_2500k]
test_2500k_frames = [aimd_2500k[index] for index in test_indices_2500k]

train_frames_all = train_500k_frames + train_1500k_frames + train_2500k_frames
validation_frames_all = validation_500k_frames + validation_1500k_frames + validation_2500k_frames
test_frames_all = test_500k_frames + test_1500k_frames + test_2500k_frames

ase.io.write('train-random-{0}frames.extxyz'.format(len(train_frames_all)), train_frames_all)
ase.io.write('validation-random-{0}frames.extxyz'.format(len(validation_frames_all)), validation_frames_all)
ase.io.write('test-random-{0}frames.extxyz'.format(len(test_frames_all)), test_frames_all)
# concatenated_trajectories = aimd_500k + aimd_1500k + aimd_2500k
# ase.io.write('concatenated-aimd.extxyz', concatenated_trajectories)