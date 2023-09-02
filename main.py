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
    parser.add_argument('-r', '--reduce', action='store_true', help='turn on model order reduction')
    args = parser.parse_args(argv)
    # print(args.network, args.input_sources, args.output_nodes, args.reduce)

    input_sources = set(args.input_sources)
    output_nodes = set(args.output_nodes)

    circuit = Circuit(args.network, input_sources, output_nodes)
    print("circuit model size:")
    circuit.print_GCb_matrices()
    (t, outputs) = transient.transient_analysis(circuit)
    for output in outputs:
        plt.plot(t, output)

    if args.reduce:
        reduced_model_order = 20   # order to reduce model down to
        tic = time.perf_counter()
        reduced_circuit = prima.PrimaReducedCircuit(reduced_model_order, circuit)
        toc = time.perf_counter()
        print("simulating the circuit took %.6f seconds" % (toc - tic))
        print("reduced circuit model size:")
        reduced_circuit.print_GCb_matrices()

        (t, outputs) = transient.transient_analysis(reduced_circuit)
        for output in outputs:
            plt.plot(t, output)

    plt.savefig("mygraph.png")


if __name__ == "__main__":
    main(sys.argv[1:])
