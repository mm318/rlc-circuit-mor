#!/usr/bin/env python3

from enum import Enum
import time
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

from . import prima

class SolverMethod(Enum):
    SOLVE = 1
    FORWARD_BACKWARD_SUBSTITUTION = 2
    INVERSE = 3

METHOD = SolverMethod.INVERSE

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

def implicit_integrate(C, G, b, B, x0, ti, tf, dt):
    # Given:
    # G*x(t) + C*x'(t) = b(t)
    # C*x'(t) = b(t) - G*x(t)
    #
    # trapezoidal rule:
    # x(t+dt) = x(t) + 0.5*dt*(x'(t+dt) + x(t))
    # C*x(t+dt) = C*x(t) + 0.5*dt*(C*x'(t+dt) + C*x(t))
    # C*x(t+dt) = C*x(t) + 0.5*dt*((b(t+dt) - G*x(t+dt)) + (b(t) - G*x(t)))
    # (C + 0.5*dt*G)*x(t+dt) = (C - 0.5*dt*G)*x(t) + 0.5*dt*(b(t+dt) + b(t))

    b = b.flatten() # constant inputs
    B = B.flatten() # time-dependent (user-defined) inputs

    A_rhs = (C - (dt/2)*G)
    A = (C + (dt/2)*G)
    if METHOD == SolverMethod.FORWARD_BACKWARD_SUBSTITUTION:
        (P, L, U) = scipy.linalg.lu(A)
        # with np.printoptions(linewidth=1000):
        #     print(P)
        #     print(L)
        #     print(U)
    elif METHOD == SolverMethod.INVERSE:
        inv_A = np.linalg.inv(A)

    num_points = math.ceil((tf - ti) / dt)

    t = np.empty(num_points+1)
    x = np.empty((x0.shape[0], num_points+1))

    t[0] = ti
    x[:, 0:1] = x0

    for i in range(num_points):
        x_curr = x[:, i]
        t_curr = ti + i*dt
        t_next = ti + (i+1)*dt
        u_curr = square_wave(t_curr)
        u_next = square_wave(t_next)
        u_avg = (u_curr + u_next) / 2

        rhs = np.matmul(A_rhs, x_curr) + dt*(b + B*u_avg)

        if METHOD == SolverMethod.SOLVE:
            x_next = np.linalg.solve(A, rhs)    # this is slower, but more numerically stable
        elif METHOD == SolverMethod.FORWARD_BACKWARD_SUBSTITUTION:
            x_next = fwd_bwd_sub(L, U, P, rhs)  # TODO: why is this numerically unstable when dt is small
        elif METHOD == SolverMethod.INVERSE:
            x_next = np.matmul(inv_A, rhs)      # this is fastest. unsure about stability

        t[i+1] = t_next
        x[:, i+1] = x_next

    return (t, x)

def transient_analysis(circuit, test_mor=False):
    (G, C, b) = circuit.mna_GCb_matrices
    print("circuit model size:")
    circuit.print_GCb_matrices()

    print('replacing voltage/current sources with square waves')
    B = circuit.input_B_vector

    x0 = np.zeros(b.shape)
    tic = time.perf_counter()
    (t, x) = implicit_integrate(C, G, b, B, x0, 0, 5e-9, 0.02e-9)
    toc = time.perf_counter()
    print("simulating the circuit took %f seconds" % (toc - tic))

    print('observing the following nodes:')
    L_list = circuit.output_L_vectors

    for L in L_list:
        output = np.dot(L.transpose(), x).flatten()
        plt.plot(t, output)
    plt.savefig("mygraph.png")

    if test_mor:
        q = 6   # order to reduce model down to
        reduced_circuit = prima.reduce(q, G, C, b, B, L_list)
