#!/usr/bin/env python3

import numpy as np


class Circuit:
    def __init__(self, filename):
        self.node_name_to_id = { '0' : -1, 'node0' : -1, 'gnd' : -1, 'ground' : -1 }
        num_default_nodes = len(self.node_name_to_id)
        self.voltage_sources = {}   # internal voltage sources (num_internals): voltage_sources[component_name] = v_src_id
        self.current_sources = {}   # internal current sources: current_sources[component_name] = (node1_id, node2_id)

        input_file = open(filename, 'r')
        components = []
        i_size = 0  # number of internal current sources
        for line in input_file:
            line = line.strip()
            if line == "" or line[0] == '#' or line[0] == '*' or line[0] == '.':
                continue;

            param_strs = line.split()
            # print(param_strs)
            assert(len(param_strs) == 4)

            component_name = param_strs[0]
            node1_name = param_strs[1]
            node2_name = param_strs[2]
            value = float(param_strs[3])

            components.append((component_name, node1_name, node2_name, value))
            if node1_name not in self.node_name_to_id:
                node_id = len(self.node_name_to_id) - num_default_nodes
                self.node_name_to_id[node1_name] = node_id
            if node2_name not in self.node_name_to_id:
                node_id = len(self.node_name_to_id) - num_default_nodes
                self.node_name_to_id[node2_name] = node_id

            component_type = component_name[0].lower()
            if component_type == 'r':
                i_size = max(i_size, self.node_name_to_id[node1_name]+1, self.node_name_to_id[node2_name]+1)
            elif component_type == 'c':
                i_size = max(i_size, self.node_name_to_id[node1_name]+1, self.node_name_to_id[node2_name]+1)
            elif component_type == 'l':
                v_src_id = len(self.voltage_sources)
                self.voltage_sources[component_name] = v_src_id
            elif component_type == 'v':
                v_src_id = len(self.voltage_sources)
                self.voltage_sources[component_name] = v_src_id
            elif component_type == 'i':
                i_size = max(i_size, self.node_name_to_id[node1_name]+1, self.node_name_to_id[node2_name]+1)
                self.current_sources[component_name] = (self.node_name_to_id[node1_name], self.node_name_to_id[node2_name])
            else:
                print('unknown component %s' % component_name)
        input_file.close()

        num_nodes = len(self.node_name_to_id) - num_default_nodes
        v_size = len(self.voltage_sources)

        # left hand side component (Gx(t) + Cx'(t) = b(t))
        self.GA = np.zeros((num_nodes, num_nodes))  # lhs voltages wrt rhs currents
        self.GB = np.zeros((num_nodes, v_size))     # lhs currents wrt rhs currents
        self.GC = np.zeros((v_size, num_nodes))     # lhs voltages wrt rhs voltages
        self.GD = np.zeros((v_size, v_size))        # lhs currents wrt rhs voltages

        # left hand side component (Gx(t) + Cx'(t) = b(t))
        self.CA = np.zeros((num_nodes, num_nodes))  # lhs voltage changes wrt rhs currents
        self.CB = np.zeros((num_nodes, v_size))     # lhs current changes wrt rhs currents
        self.CC = np.zeros((v_size, num_nodes))     # lhs voltage changes wrt rhs voltages
        self.CD = np.zeros((v_size, v_size))        # lhs current changes wrt rhs voltages

        # right hand side component (non-user inputs)
        self.i = np.zeros((num_nodes, 1))   # fixed currents (not states)
        self.v = np.zeros((v_size, 1))      # fixed voltages (not states)
        # print(i_size, num_nodes)

        for (component_name, node1_name, node2_name, value) in components:
            component_type = component_name[0].lower()
            if component_type == 'r':
                self._add_resistor(self.node_name_to_id[node1_name], self.node_name_to_id[node2_name], value)
            elif component_type == 'c':
                self._add_capacitor(self.node_name_to_id[node1_name], self.node_name_to_id[node2_name], value)
            elif component_type == 'l':
                self._add_inductor(component_name, self.node_name_to_id[node1_name], self.node_name_to_id[node2_name], value)
            elif component_type == 'v':
                self._add_voltage_source(component_name, self.node_name_to_id[node1_name], self.node_name_to_id[node2_name], value)
            elif component_type == 'i':
                self._add_current_source(self.node_name_to_id[node1_name], self.node_name_to_id[node2_name], value)

        self.G = np.vstack((np.hstack((self.GA, self.GB)), np.hstack((self.GC, self.GD))))
        self.C = np.vstack((np.hstack((self.CA, self.CB)), np.hstack((self.CC, self.CD))))
        self.b = np.vstack((self.i, self.v))
        assert(self.G.shape[0] == self.b.shape[0])
        assert(self.G.shape[1] == self.b.shape[0])

        self.G.setflags(write=False)
        self.C.setflags(write=False)
        self.b.setflags(write=False)

    def _add_resistor(self, node1_id, node2_id, value):
        if node1_id >= 0:
            self.GA[node1_id, node1_id] += 1/value
        if node2_id >= 0:
            self.GA[node2_id, node2_id] += 1/value
        if node1_id >= 0 and node2_id >= 0:
            self.GA[node1_id, node2_id] -= 1/value
            self.GA[node2_id, node1_id] -= 1/value

    def _add_capacitor(self, node1_id, node2_id, value):
        if node1_id >= 0:
            self.CA[node1_id, node1_id] += value
        if node2_id >= 0:
            self.CA[node2_id, node2_id] += value
        if node1_id >= 0 and node2_id >= 0:
            self.CA[node1_id, node2_id] -= value
            self.CA[node2_id, node1_id] -= value

    def _add_inductor(self, component_name, node1_id, node2_id, value):
        assert(node1_id >= 0 or node2_id >= 0)
        v_src_id = self.voltage_sources[component_name]
        
        self.CD[v_src_id, v_src_id] = value

        # current flowing into and out of inductor
        if node1_id >= 0:
            self.GB[node1_id, v_src_id] += 1
        if node2_id >= 0:
            self.GB[node2_id, v_src_id] -= 1

        # voltage drops across inductor
        if node1_id >= 0:
            self.GC[v_src_id, node1_id] -= 1
        if node2_id >= 0:
            self.GC[v_src_id, node2_id] += 1

    def _add_voltage_source(self, component_name, p_node_id, n_node_id, value):
        assert(p_node_id != n_node_id)
        v_src_id = self.voltage_sources[component_name]

        self.v[v_src_id, 0] = value # fixed component voltage drop across voltage source

        # current relationships (KCL)
        if p_node_id >= 0:
            self.GB[p_node_id, v_src_id] -= 1
        if n_node_id >= 0:
            self.GB[n_node_id, v_src_id] += 1

        # voltage relationships (KVL)
        if p_node_id >= 0:
            self.GC[v_src_id, p_node_id] += 1
        if n_node_id >= 0:
            self.GC[v_src_id, n_node_id] -= 1

    def _add_current_source(self, node1_id, node2_id, value):
        if node1_id >= 0:
            self.i[node1_id] -= value
        if node2_id >= 0:
            self.i[node2_id] += value

    @property
    def mna_matrices(self):
        return (self.G, self.C, np.copy(self.b))

    def voltage_source_bvec_idx(self, name):
        if name not in self.voltage_sources:
            print('voltage source %s not found in circuit' % name)
            return None
        v_src_id = self.voltage_sources[name]
        return self.i.shape[0] + v_src_id

    def current_source_bvec_idxs(self, name):
        if name not in self.current_sources:
            print('current source %s not found in circuit' % name)
            return None
        return self.current_sources[name]

    def node_xvec_idx(self, name):
        if name not in self.node_name_to_id:
            print('node %s not found in circuit' % name)
            return None
        return self.node_name_to_id[name]

    def print_matrices(self):
        with np.printoptions(linewidth=1000):
            print('G(%s) =\n' % str(self.G.shape), self.G)
            print('C(%s) =\n' % str(self.C.shape), self.C)
            print('b(%s) =\n' % str(self.b.shape), self.b)
