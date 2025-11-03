import numpy as np
import ase
from ase.io import read
import dscribe
from dscribe.descriptors import SOAP, CoulombMatrix, ACSF, MBTR
from vendi_score import vendi
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Assume soap_descriptors is an (N, D) NumPy array
def rbf_similarity_matrix(descriptors, gamma):
    N = descriptors.shape[0]
    # Compute squared Euclidean distance matrix
    diff = descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Apply RBF kernel
    K = np.exp(-gamma * dist_sq)
    return K

def vary_nframes(nframes_list):
    for j in nframes_list:
        # i = 5000-j
        # traj500 = ase.io.read('/home/cganley2/qe-data/sic-3c/simulations/aimd-2500K/SiC_3C-aimd-2500K.out.extxyz', index='{0}:'.format(j))
        traj500 = ase.io.read('concatenated-aimd.extxyz', index=':')
        i = len(traj500)

        mbtr_desc = MBTR(
            species=["Si", "C"],
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            periodic=True,
            normalization="l2",
        )

        mbtr_all = np.vstack([mbtr_desc.create(traj500[index], n_jobs=8) for index, value in enumerate(traj500)])

        gamma = 5e5
        mbtr_sim = rbf_similarity_matrix(mbtr_all, gamma)

        with open('MBTR-RBF-RL300-{0}frames-sim-matrix.pkl'.format(i), 'wb') as f:
            pickle.dump(mbtr_sim, f)

        vs = vendi.score_K(mbtr_sim)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 2 rows, 1 column layout

        # Plot heatmap for mbtr_sim
        im1 = ax1.imshow(mbtr_sim, cmap='viridis', aspect='auto')
        ax1.set_title('MBTR RBF Similarity', fontsize=20)
        ax1.set_xlabel('Frame #')
        ax1.tick_params(axis='x', labelsize=20)#, rotation=45)
        ax1.tick_params(axis='y', labelsize=20)
        plt.colorbar(im1, ax=ax1)

        # Add label "A" to top-left corner of ax1
        ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

        values = mbtr_sim[np.triu_indices_from(mbtr_sim, k=1)]
        sns.histplot(values, bins=30, kde=False) # Histogram with density
        ax2.set_xlabel('Similarity Score', fontsize=20)
        ax2.set_title(r'Similarity Distribution: $\sigma$=0.003', fontsize=20)
        ax2.set_yticklabels([])
        ax2.set_ylabel('Count', fontsize=20)
        # ax2.text(0.35, 6000, 'VS = {0}'.format(vs), fontsize=18)
        ax2.tick_params(axis='x', labelsize=20)


        ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

        fig.suptitle('VS = {0}'.format(vs), fontsize=16)

        # plt.tight_layout()
        # plt.show()
        fig.savefig('MBTR-RBF-RL300-{0}frames-Aheatmap-Bhist.png'.format(i), dpi=600) #, bbox_inches='tight')

        with open('MBTR-RBF-RL300-{0}frames-Aheatmap-Bhist.pkl'.format(i), 'wb') as f:
            pickle.dump(fig, f)

def vary_gamma(gamma_list):
    for j in gamma_list:
        # i = 5000-j
        traj500 = ase.io.read('train-rbf-1200frames.extxyz', index=':')

        mbtr_desc = MBTR(
            species=["Si", "C"],
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            periodic=True,
            normalization="l2",
        )

        mbtr_all = np.vstack([mbtr_desc.create(traj500[index], n_jobs=32) for index, value in enumerate(traj500)])

        gamma = j
        mbtr_sim = rbf_similarity_matrix(mbtr_all, gamma)

        with open('MBTR-RBF-{0}gamma-{1}frames-sim-matrix.pkl'.format(int(gamma), len(traj500)), 'wb') as f:
            pickle.dump(mbtr_sim, f)

        vs = vendi.score_K(mbtr_sim)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # 2 rows, 1 column layout

        # Plot heatmap for mbtr_sim
        im1 = ax1.imshow(mbtr_sim, cmap='viridis', aspect='auto')
        ax1.set_title('MBTR RBF Similarity', fontsize=20)
        plt.colorbar(im1, ax=ax1)

        # Add label "A" to top-left corner of ax1
        ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

        values = mbtr_sim[np.triu_indices_from(mbtr_sim, k=1)]
        sns.histplot(values, bins=30, kde=False) # Histogram with density
        ax2.set_xlabel('Similarity Score', fontsize=20)
        ax2.set_title(r'Similarity Distribution: $\gamma$={0}'.format(int(gamma)), fontsize=20)
        ax2.set_yticklabels([])
        ax2.set_ylabel('Count', fontsize=20)
        # ax2.text(0.35, 6000, 'VS = {0}'.format(vs), fontsize=18)
        ax2.tick_params(axis='x', labelsize=20)


        ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

        # fig.suptitle('VS = {0}'.format(vs), fontsize=16)
        np.save('gamma{0}-vs{1}'.format(int(gamma), vs), vs)

        # plt.tight_layout()
        # plt.show()
        fig.savefig('MBTR-RBF-{0}gamma-Aheatmap-Bhist.png'.format(int(gamma)), dpi=400) #, bbox_inches='tight')

        with open('MBTR-RBF-{0}gamma-Aheatmap-Bhist.pkl'.format(int(gamma)), 'wb') as f:
            pickle.dump(fig, f)

if __name__ == '__main__':
    # vary_nframes([300])
    vary_gamma([1e6])
