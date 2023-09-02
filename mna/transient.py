#!/usr/bin/env python3

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt


def square_wave(t):
    MAX_VALUE = 1
    PERIOD_S = 2e-9
    RISING_FALLING_TIME_S = 0.1e-9

    # period of 2 nanoseconds
    periodic_pos = (t - RISING_FALLING_TIME_S) % PERIOD_S
    assert(periodic_pos >= 0)

    # rise/fall time is 0.1 ns
    if 0 <= periodic_pos and periodic_pos <= RISING_FALLING_TIME_S:
        u = MAX_VALUE * (periodic_pos / RISING_FALLING_TIME_S)
    elif RISING_FALLING_TIME_S <= periodic_pos and periodic_pos <= PERIOD_S/2:
        u = MAX_VALUE
    elif PERIOD_S/2 <= periodic_pos and periodic_pos <= PERIOD_S/2+RISING_FALLING_TIME_S:
        u = MAX_VALUE - ((periodic_pos - PERIOD_S/2) / RISING_FALLING_TIME_S)
    else:
        u = 0

    return u

def fwd_bwd_sub(L, U, P, b):
    b = np.dot(P, b)

    # initializing forward elimination
    y = np.zeros(b.shape)
    y[0] = b[0] / L[0, 0]
    # forward elimination
    n = L.shape[0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]

    # initializing backward substitution
    x = np.zeros(y.shape)
    x[-1] = y[-1] / U[-1, -1]
    # backward substitution
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]

    return x

def implicit_integrate(C, G, b, bvec_src_idxs, x0, ti, tf, dt):
    # Given:
    # G*x(t) + C*x'(t) = b(t)
    # C*x'(t) = b(t) - G*x(t)
    #
    # trapezoidal rule:
    # x(t+dt) = x(t) + 0.5*dt*(x'(t+dt) + x(t))
    # C*x(t+dt) = C*x(t) + 0.5*dt*(C*x'(t+dt) + C*x(t))
    # C*x(t+dt) = C*x(t) + 0.5*dt*((b(t+dt) - G*x(t+dt)) + (b(t) - G*x(t)))
    # (C + 0.5*dt*G)*x(t+dt) = (C - 0.5*dt*G)*x(t) + 0.5*dt*(b(t+dt) + b(t))

    b = b.reshape(b.shape[0])
    A_rhs = (C - (dt/2)*G)
    A = (C + (dt/2)*G)
    (P, L, U) = scipy.linalg.lu(A)

    num_points = math.ceil((tf - ti) / dt)

    t = np.empty(num_points+1)
    x = np.empty((x0.shape[0], num_points+1))

    t[0] = ti
    x[:, 0] = x0.reshape(x0.shape[0])

    for i in range(num_points):
        x_curr = x[:, 1]
        t_curr = ti + i*dt
        t_next = ti + (i+1)*dt
        u_curr = square_wave(t_curr)
        u_next = square_wave(t_next)
        u_avg = (u_curr + u_next) / 2

        b[bvec_src_idxs[0]] = u_avg
        b[bvec_src_idxs[1]] = -u_avg
        rhs = np.matmul(A_rhs, x_curr) + dt*b
        x_next = fwd_bwd_sub(L, U, P, rhs)

        t[i+1] = t_next
        x[:, i+1] = x_next.reshape(x_next.shape[0])

    return (t, x)

def transient_analysis(circuit, input_component_names, output_node_names):
    (G, C, b) = circuit.mna_matrices

    print('replacing the following voltage/current sources with square waves:')
    pos_bvec_idxs = set()
    neg_bvec_idxs = set()
    for src_name in input_component_names:
        component_type = src_name[0].lower()
        if component_type == 'i':
            indices = circuit.current_source_bvec_idxs(src_name)
            print('  %s [b_vector indices=%s]' % (src_name, str(indices)))
            neg_bvec_idxs.add(indices[0])
            pos_bvec_idxs.add(indices[1])
        elif component_type == 'v':
            index = circuit.voltage_source_bvec_idx(src_name)
            print('  %s [b_vector index=%d]' % (src_name, index))
            pos_bvec_idxs.add(index)
        else:
            print('  %s (invalid source)' % src_name)
    pos_bvec_idxs = list(pos_bvec_idxs)
    neg_bvec_idxs = list(neg_bvec_idxs)

    x0 = np.zeros(b.shape)
    # implicit_integrate(C, G, b, (pos_bvec_idxs, neg_bvec_idxs), x0, 0, 5e-9, 100e-12)

    print('observing the following nodes:')
    for node_name in output_node_names:
        index = circuit.node_xvec_idx(node_name)
        print('  %s [x_vector_index=%d]' % (node_name, index))

    t = np.linspace(0, 10e-9, 300)
    vsquare_wave = np.vectorize(square_wave)
    u = vsquare_wave(t)
    plt.plot(t, u)
    plt.savefig("mygraph.png")

