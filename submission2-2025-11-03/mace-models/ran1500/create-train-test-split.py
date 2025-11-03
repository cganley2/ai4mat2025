import numpy as np
import ase
from ase import io

train_500K_500indices = np.random.randint(0, 5000, 500)
train_1500K_500indices = np.random.randint(0, 5000, 500)
train_2500K_500indices = np.random.randint(0, 4999, 500)

aimd500 = ase.io.read('/scratch/cganley2/mlip/Maximize_Diversity_using_RL_Dataset/Diverse_Subset_from_500K/SiC_3C-aimd-500K.out', index=':')
aimd1500 = ase.io.read('/scratch/cganley2/mlip/Maximize_Diversity_using_RL_Dataset/Diverse_Subset_from_1500K/SiC_3C-aimd-1500K.out', index=':')
aimd2500 = ase.io.read('/scratch/cganley2/mlip/Maximize_Diversity_using_RL_Dataset/Diverse_Subset_from_2500K/SiC_3C-aimd-2500K.out', index=':')


training_500K_500frames = [aimd500[index] for index in train_500K_500indices]
training_1500K_500frames = [aimd1500[index] for index in train_1500K_500indices]
training_2500K_500frames = [aimd2500[index] for index in train_2500K_500indices]

training_1500 = training_500K_500frames + training_1500K_500frames + training_2500K_500frames

test_500K_4500_frames = [aimd500[index] for index, value in enumerate(aimd500) if index not in train_500K_500indices]
test_1500K_4500_frames = [aimd1500[index] for index, value in enumerate(aimd1500) if index not in train_1500K_500indices]
test_2500K_4500_frames = [aimd2500[index] for index, value in enumerate(aimd2500) if index not in train_2500K_500indices]

test_13500 = test_500K_4500_frames + test_1500K_4500_frames + test_2500K_4500_frames

ase.io.write('ran-train1500.extxyz', training_1500)
ase.io.write('ran-test13500.extxyz', test_13500)