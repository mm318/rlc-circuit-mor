#!/usr/bin/env python3

import time
from multiprocessing import Pool
import numpy as np


def transfer_function(G, C, B, s):
    # Given:
    # G*x(t) + C*x'(t) = b + B*u(t)
    # y(t) = L'*x(t)
    # [b is a time-constant vector, u(t) is a scalar]
    # [y(t) is the desired observable, x(t) are the internal states of the model]
    #
    # find transfer function:
    # G*x(s) + s*C*x(s) = b/s + B*u(s)
    # x(s) = (G + s*C)^-1 * (b/s + B*u(s))
    # [assuming passive circuit, b = 0]
    # x(s)/u(s) = (G + s*C)^-1 * B
    #
    # y(s) = L'*x(s)
    #
    # H(s) = y(s)/u(s) = L'*x(s)/u(s) = L'*(G + s*C)^-1 * B
    A = (G + s*C)
    return np.linalg.solve(A, B)

def frequency_analysis(circuit, w_lo, w_hi):
    (G, C, b) = circuit.mna_GCb_matrices
    B = circuit.input_B_vector
    L_list = circuit.output_L_vectors
    if len(circuit.internal_source_names) > 0:
        print('assuming internal sources are zero (passive circuit)')
        for component_name in circuit.internal_source_names:
            print('  %s' % component_name)

    w = np.logspace(w_lo, w_hi, 500)
    tic = time.perf_counter()
    with Pool(processes=8) as pool:
        computing = []
        for wi in w:
            result = pool.apply_async(transfer_function, (G, C, B, 1j*wi))
            computing.append(result)

        results = []
        for promise in computing:
            results.append(promise.get())

    outputs = []
    for (node_name, L) in zip(circuit.output_node_names, L_list):
        output = np.empty(w.shape, dtype=complex)
        assert len(output) == len(results)
        for i in range(len(results)):
            output[i] = np.dot(L.transpose(), results[i])[0][0]
        outputs.append((node_name, output))
    toc = time.perf_counter()
    print("analyzing the circuit took %.6f seconds" % (toc - tic))

    return (w, outputs)
