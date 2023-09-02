#!/usr/bin/env python3

import sys
import argparse

from mna.circuit import Circuit
from mna import transient


def main(argv):
    parser = argparse.ArgumentParser(description='Run modified nodal analysis on a given network.')
    parser.add_argument('network', metavar='N', type=str, help='filename of circuit (SPICE format)')
    parser.add_argument('-i', '--input_sources', metavar='I', type=str, required=True, nargs='+', help='component name(s) of circuit inputs')
    parser.add_argument('-o', '--output_nodes', metavar='O', type=str, required=True, nargs='+', help='node name(s) of circuit to observe')
    parser.add_argument('-r', '--reduce', action='store_true', help='turn on model order reduction')
    args = parser.parse_args(argv)
    # print(args.network, args.input_sources, args.output_nodes, args.reduce)

    circuit = Circuit(args.network)
    circuit.print_matrices()

    transient.transient_analysis(circuit, args.input_sources, args.output_nodes)


if __name__ == "__main__":
    main(sys.argv[1:])
