# -*- coding: utf-8 -*-
# @Time : 2022/12/9 16:42
# @Author : Xiao Han
# @E-mail : hahahenha@gmail.com
# @Site : 
# @project: spectral gnn
# @File : Preprocessing.py
# @Software: PyCharm
import math

import networkx as nx
import numpy as np
from scipy import sparse
from utils.preprocess.filters import heat, approximations, self_define
from utils.preprocess.graphs import graph
from sklearn.preprocessing import normalize

class Preprocessing(object):
    def __init__(self, g, scale=1, approximation_order=3, tolerance=0.0, model_name="HGNN", data_name="job"):
        self.graph = g
        self.data_name = data_name
        self.processing_graph = graph.Graph(nx.adjacency_matrix(self.graph))
        self.processing_graph.estimate_lmax()
        # print('Lmax:{:.8f}'.format(self.processing_graph.lmax))
        if model_name == 'HGNN':
            self.scales = [-1.0/scale, 1.0/scale]
            while(scale - 1 > 0):
                sc = int(scale-1)
                self.scales.append(-1.0/sc)
                self.scales.append(1.0/sc)
                scale = sc
        else:
            self.scales = [-1.0/scale, 1.0/scale]
        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.phi_matrices = []

    def calculate_wavelet(self):
        impulse = np.eye(self.graph.number_of_nodes(), dtype=int)

        wavelet_coefficients = approximations.cheby_op(self.processing_graph,
                                                       self.chebyshev,
                                                       impulse)
        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        # print('wavelet_coefficients shape:', wavelet_coefficients.shape)
        ind_1, ind_2 = wavelet_coefficients.nonzero()
        n_count = self.graph.number_of_nodes()
        remaining_waves = sparse.csr_matrix((wavelet_coefficients[ind_1, ind_2], (ind_1, ind_2)),
                                            shape=(n_count, n_count),
                                            dtype=np.float32)
        return remaining_waves

    def normalize_matrices(self):
        # print("\nNormalizing the sparsified wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            self.phi_matrices[i] = normalize(self.phi_matrices[i], norm='l1', axis=1)

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = len(self.phi_matrices[0].nonzero()[0])/(self.graph.number_of_nodes()**2)
        wavelet_density = str(round(100*wavelet_density, 2))
        inverse_wavelet_density = len(self.phi_matrices[1].nonzero()[0])/(self.graph.number_of_nodes()**2)
        inverse_wavelet_density = str(round(100*inverse_wavelet_density, 2))
        # print("\tDensity of wavelets: "+wavelet_density+"%.")
        # print("\tDensity of inverse wavelets: "+inverse_wavelet_density+"%.\n")

    def calculate_all(self):
        # print("\nProprecessing started.\n")
        for i, scale in enumerate(self.scales):
            if self.data_name == 'citeseer':
                self.heat_filter = self_define.Self_define(self.processing_graph,tau=[scale])
            else:
                self.heat_filter = heat.Heat(self.processing_graph,tau=[scale])
            self.chebyshev = approximations.compute_cheby_coeff(self.heat_filter,m=self.approximation_order)
            sparsified_wavelets = self.calculate_wavelet()
            self.phi_matrices.append(sparsified_wavelets)
        self.normalize_matrices()
        self.calculate_density()
