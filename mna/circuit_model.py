#!/usr/bin/env python3

from abc import ABC, abstractmethod

class CircuitModel(ABC):
    @property
    @abstractmethod
    def mna_GCb_matrices(self):
        pass

    @property
    @abstractmethod
    def input_B_vector(self):
        pass

    @property
    @abstractmethod
    def output_L_vectors(self):
        pass

    @property
    @abstractmethod
    def internal_source_names(self):
        pass

    @property
    @abstractmethod
    def output_node_names(self):
        pass

    @abstractmethod
    def print_GCb_matrices(self):
        pass
