#!/usr/bin/env python3

import numpy as np

from .circuit_model import CircuitModel


class PrimaReducedCircuit(CircuitModel):
    def __init__(self, q, full_circuit):
        (G, C, b) = full_circuit.mna_GCb_matrices
        B = full_circuit.input_B_vector
        L_list = full_circuit.output_L_vectors

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
        self.Gq = Vq.transpose() @ G @ Vq
        self.Cq = Vq.transpose() @ C @ Vq
        self.bq = Vq.transpose() @ b
        self.Bq = Vq.transpose() @ B
        self.Lq_list = []
        for L in L_list:
            Lq = Vq.transpose() @ L
            Lq.setflags(write=False)
            self.Lq_list.append(Lq)

        self.Gq.setflags(write=False)
        self.Cq.setflags(write=False)
        self.bq.setflags(write=False)
        self.Bq.setflags(write=False)

    @property
    def mna_GCb_matrices(self):
        return (self.Gq, self.Cq, self.bq)

    @property
    def input_B_vector(self):
        return self.Bq

    @property
    def output_L_vectors(self):
        return self.Lq_list

    def print_GCb_matrices(self):
        with np.printoptions(linewidth=1000):
            print('G(%s) =\n' % str(self.Gq.shape), self.Gq)
            print('C(%s) =\n' % str(self.Cq.shape), self.Cq)
            print('b(%s) =\n' % str(self.bq.shape), self.bq)
