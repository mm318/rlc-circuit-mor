#!/usr/bin/env python3

import numpy as np

from .circuit_model import CircuitModel


class ReducedCircuit(CircuitModel):
    @property
    def mna_GCb_matrices(self):
        pass

    @property
    def input_B_vector(self):
        pass

    @property
    def output_L_vectors(self):
        pass

    def print_GCb_matrices(self):
        pass


def reduce(q, G, C, b, B, L_list):
    n = G.shape[0]  # Order of original system

    R = np.linalg.solve(G, b+B)
    (Q, X) = np.linalg.qr(R)

    # Generate first block V_0 of projection matrix
    Vq = np.zeros((n, q+1))
    Vq[:, 0:1] = Q

    # Arnoldi iteration
    for j in range(1, q):
        Vq[:,j] = -np.linalg.solve(G, np.matmul(C, Vq[:,j-1]))
        # Modified Gram-Schmidt orthonormalization
        for i in range(j):
            delta = np.matmul(Vq[:,i].transpose(), Vq[:,j])
            Vq[:,j] = Vq[:,j] - delta*Vq[:,i]
        (Q, X) = np.linalg.qr(Vq[:,j:j+1])
        Vq[:,j:j+1] = Q
    Vq = Vq[:, 0:q]
    # print(Vq.shape)

    # Matrices projection
    Gq = Vq.transpose() @ G @ Vq
    Cq = Vq.transpose() @ C @ Vq
    bq = Vq.transpose() @ b
    Bq = Vq.transpose() @ B
    Lq_list = []
    for L in L_list:
        Lq = Vq.transpose() @ L
        Lq_list.append(Lq)
