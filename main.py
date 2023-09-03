#!/usr/bin/env python3

import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mna.circuit import Circuit
from mna import prima
from mna import transient
from mna import frequency


def analyze_transient(circuit, reduced_circuit=None):
    print('[starting transient analysis]')

    # transient simulation parameters
    ti = 0      # 0 ns
    tf = 7e-9   # 7 ns

    plt.figure(dpi=1200)

    (full_t, full_outputs) = transient.transient_analysis(circuit, ti, tf)
    if reduced_circuit is not None:
        (reduced_t, reduced_outputs) = transient.transient_analysis(reduced_circuit, ti, tf)
        for (node_name, output) in reduced_outputs:
            plt.plot(reduced_t, output, label="reduced circuit node %s" % node_name, linewidth=0.5)
    for (node_name, output) in full_outputs:
        plt.plot(full_t, output, label="full circuit node %s" % node_name, linewidth=0.3)

    plt.title('Voltage vs Time')
    plt.xlabel('time (s)')
    plt.xlim(ti, tf)
    plt.ylabel('voltage (V)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("transient_analysis.png")
    plt.close()
    print('[finished transient analysis]')

def analyze_frequency(circuit, reduced_circuit=None):
    print('[starting frequency analysis]')
    # frequency analysis parameters
    w_lo = -1
    w_hi = 19

    fig = plt.figure(dpi=1200)
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)

    (full_w, full_outputs) = frequency.frequency_analysis(circuit, w_lo, w_hi)
    line_handles = []
    if reduced_circuit is not None:
        (reduced_w, reduced_outputs) = frequency.frequency_analysis(reduced_circuit, w_lo, w_hi)
        for (node_name, output) in reduced_outputs:
            line = ax0.plot(reduced_w, np.real(output), label="reduced circuit node %s" % node_name, linewidth=0.5)
            ax1.plot(reduced_w, np.imag(output), label="reduced circuit node %s" % node_name, linewidth=0.5)
            line_handles.append(line[0])
    for (node_name, output) in full_outputs:
        line = ax0.plot(full_w, np.real(output), label="full circuit node %s" % node_name, linewidth=0.3)
        ax1.plot(full_w, np.imag(output), label="full circuit node %s" % node_name, linewidth=0.3)
        line_handles.append(line[0])

    fig.suptitle('Frequency Response vs Frequency')

    ax0.set_xlabel('s (rad/s)')
    ax0.set_xlim(10**w_lo, 10**w_hi)
    ax0.set_xscale('log')
    ax0.set_ylabel('Re(H(s))')

    ax1.set_xlabel('s (rad/s)')
    ax1.set_xlim(10**w_lo, 10**w_hi)
    ax1.set_xscale('log')
    ax1.set_ylabel('Im(H(s))')

    fig.legend(handles=line_handles, loc='upper right', fontsize='x-small', borderpad=0.2)
    plt.tight_layout()
    fig.savefig("frequency_analysis.png")
    plt.close()
    print('[finished frequency analysis]')

def main(argv):
    parser = argparse.ArgumentParser(description='Run modified nodal analysis on a given network.')
    parser.add_argument('network', metavar='N', type=str, help='filename of circuit (SPICE format)')
    parser.add_argument('-i', '--input_sources', metavar='I', type=str, required=True, nargs='+', help='component name(s) of circuit inputs')
    parser.add_argument('-o', '--output_nodes', metavar='O', type=str, required=True, nargs='+', help='node name(s) of circuit to observe')
    parser.add_argument('-r', '--reduce', metavar='R', type=int, nargs=1, help='experiment with model order reduction using given order')
    args = parser.parse_args(argv)

    input_sources = set(args.input_sources)
    watch_nodes = set(args.output_nodes)

    circuit = Circuit(args.network, input_sources, watch_nodes)
    print("circuit model size:")
    circuit.print_GCb_matrices()

    reduced_circuit = None
    if args.reduce is not None:
        tic = time.perf_counter()
        reduced_circuit = prima.PrimaReducedCircuit(args.reduce[0], circuit)
        toc = time.perf_counter()
        print("reducing the circuit model took %.6f seconds" % (toc - tic))
        print("reduced circuit model size:")
        reduced_circuit.print_GCb_matrices()

    analyze_transient(circuit, reduced_circuit)
    analyze_frequency(circuit, reduced_circuit)

if __name__ == "__main__":
    main(sys.argv[1:])
