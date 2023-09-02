#!/usr/bin/env python3

import sys
import time
import argparse
import matplotlib.pyplot as plt

from mna.circuit import Circuit
from mna import transient
from mna import prima


def main(argv):
    parser = argparse.ArgumentParser(description='Run modified nodal analysis on a given network.')
    parser.add_argument('network', metavar='N', type=str, help='filename of circuit (SPICE format)')
    parser.add_argument('-i', '--input_sources', metavar='I', type=str, required=True, nargs='+', help='component name(s) of circuit inputs')
    parser.add_argument('-o', '--output_nodes', metavar='O', type=str, required=True, nargs='+', help='node name(s) of circuit to observe')
    parser.add_argument('-r', '--reduce', metavar='R', type=int, nargs=1, help='turn on model order reduction')
    args = parser.parse_args(argv)

    input_sources = set(args.input_sources)
    watch_nodes = set(args.output_nodes)

    # transient simulation parameters
    ti = 0      # 0 ns
    tf = 7e-9   # 7 ns

    plt.figure(dpi=1200)

    circuit = Circuit(args.network, input_sources, watch_nodes)
    print("circuit model size:")
    circuit.print_GCb_matrices()
    (t, full_outputs) = transient.transient_analysis(circuit, ti, tf)

    if args.reduce is not None:
        tic = time.perf_counter()
        reduced_circuit = prima.PrimaReducedCircuit(args.reduce[0], circuit)
        toc = time.perf_counter()
        print("simulating the circuit took %.6f seconds" % (toc - tic))
        print("reduced circuit model size:")
        reduced_circuit.print_GCb_matrices()

        (t, reduced_outputs) = transient.transient_analysis(reduced_circuit, ti, tf)
        for (node_name, output) in reduced_outputs:
            plt.plot(t, output, label="reduced circuit node %s" % node_name, linewidth=0.5)
    for (node_name, output) in full_outputs:
        plt.plot(t, output, label="full circuit node %s" % node_name, linewidth=0.3)

    plt.title('Voltage vs Time')
    plt.xlabel('time (s)')
    plt.xlim(ti, tf)
    plt.ylabel('voltage (V)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("transient_analysis.png")


if __name__ == "__main__":
    main(sys.argv[1:])
